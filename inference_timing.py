"""Synchronized wall-clock timing for the existing Wan inference pipeline.

The DiT metric sums ``pipe.model_fn`` calls, including both CFG branches. Wan
calls DiT components directly, so a forward hook on ``pipe.dit`` cannot measure
this work. VAE timing covers its public ``decode`` call, including transfers and
CPU tile blending, while excluding image encoding and explicit model loading.
"""

from functools import wraps
from time import perf_counter


_MISSING = object()
_WRAPPER_OWNER = "_wan_inference_timer_owner"


class WanInferenceTimer:
    """Temporarily measure one pipeline invocation on CUDA or CPU.

    Use ``with timer: video = pipe(...)`` and read ``timer.metrics()`` after
    leaving the context. Every CUDA timing boundary synchronizes ``pipe.device``;
    CPU timing needs no synchronization. Measurements are elapsed wall-clock
    seconds, not kernel-only time. Calls which raise still contribute to the
    counters and elapsed time, and all wrapped attributes are restored.

    The context can be reused and resets its measurements on each entry. Nested
    contexts on this timer, or another timer sharing its pipeline/VAE, are not
    supported. It must not be used concurrently with another pipeline invocation.
    When disabled, it does not patch methods or synchronize and returns ``{}``.
    """

    def __init__(self, pipe, enabled=True):
        self.pipe = pipe
        self.enabled = enabled
        self._active = False
        self._restorations = []
        self._device = None
        self._device_type = None
        self._reset()

    def _reset(self):
        self.dit_seconds = 0.0
        self.dit_calls = 0
        self.vae_decode_seconds = 0.0
        self.vae_decode_calls = 0
        self.pipeline_seconds = 0.0

    def _synchronize(self):
        if self._device_type == "cuda":
            # Keep CPU use and the unit tests independent of a PyTorch install.
            import torch

            torch.cuda.synchronize(self._device)

    def _wrap(self, owner, name, seconds_name, calls_name):
        original = getattr(owner, name)
        if not callable(original):
            raise TypeError(f"{name} must be callable for inference timing")
        if getattr(original, _WRAPPER_OWNER, None) is not None:
            raise RuntimeError(f"{name} is already wrapped by an active WanInferenceTimer")

        @wraps(original)
        def measured(*args, **kwargs):
            self._synchronize()
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                try:
                    self._synchronize()
                finally:
                    elapsed = perf_counter() - started
                    setattr(self, seconds_name, getattr(self, seconds_name) + elapsed)
                    setattr(self, calls_name, getattr(self, calls_name) + 1)

        setattr(measured, _WRAPPER_OWNER, self)
        # A bound class method must be restored by removing our instance
        # override. Reassigning its bound method would leave an extra override.
        previous = vars(owner).get(name, _MISSING)
        self._restorations.append((owner, name, previous))
        setattr(owner, name, measured)

    def _restore(self):
        while self._restorations:
            owner, name, previous = self._restorations.pop()
            if previous is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, previous)

    def __enter__(self):
        if self._active:
            raise RuntimeError("WanInferenceTimer contexts cannot be nested")
        self._reset()
        self._active = True
        if not self.enabled:
            return self

        try:
            self._device = self.pipe.device
            self._device_type = getattr(self._device, "type", None) or str(self._device).split(":", 1)[0]
            if self._device_type not in {"cuda", "cpu"}:
                raise ValueError("WanInferenceTimer supports only CUDA and CPU devices")
            self._wrap(self.pipe, "model_fn", "dit_seconds", "dit_calls")
            self._wrap(self.pipe.vae, "decode", "vae_decode_seconds", "vae_decode_calls")
            self._synchronize()
            self._started = perf_counter()
        except BaseException:
            self._restore()
            self._active = False
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.enabled:
                try:
                    self._synchronize()
                finally:
                    self.pipeline_seconds = perf_counter() - self._started
        finally:
            try:
                self._restore()
            finally:
                self._active = False
        return False

    def metrics(self):
        """Return measurements in seconds for the latest context invocation."""
        if not self.enabled:
            return {}
        return {
            "dit_seconds": self.dit_seconds,
            "dit_calls": self.dit_calls,
            "vae_decode_seconds": self.vae_decode_seconds,
            "vae_decode_calls": self.vae_decode_calls,
            "pipeline_seconds": self.pipeline_seconds,
            "dit_mean_call_seconds": self.dit_seconds / self.dit_calls if self.dit_calls else 0.0,
            "other_pipeline_seconds": max(
                0.0, self.pipeline_seconds - self.dit_seconds - self.vae_decode_seconds
            ),
        }
