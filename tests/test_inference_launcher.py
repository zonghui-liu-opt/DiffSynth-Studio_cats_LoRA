import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("exit_code", [0, 23])
def test_launcher_forwards_local_paths_and_offline_mode_and_preserves_failure(tmp_path, exit_code):
    repo = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    for name in ("infer_batch_lora.sh", "infer_batch.sh"):
        shutil.copy(repo / name, checkout / name)
    model = tmp_path / "local model"
    (model / "google/umt5-xxl").mkdir(parents=True)
    for name in ("diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth"):
        (model / name).touch()
    data = checkout / "testsets"
    data.mkdir()
    (data / "metadata_6cases_480x832.csv").touch()
    lora = tmp_path / "rank64.safetensors"
    lora.touch()
    output = tmp_path / "output with spaces"
    # Test the shell end-to-end while substituting only the expensive Python workload.
    (checkout / "infer_batch.py").write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['OUTPUT_DIR'], 'observed.json').write_text(json.dumps(dict(os.environ)))\n"
        "print('stdout evidence', flush=True)\n"
        "print('stderr evidence', file=sys.stderr, flush=True)\n"
        f"sys.exit({exit_code})\n"
    )
    env = {key: value for key, value in os.environ.items() if key not in {
        "BASE_MODEL_ROOT", "DATA_ROOT", "METADATA_PATH", "LOG_PATH", "EXPERIMENT_ROOT", "COMPARE_ROOT", "EXPECTED_LORA_RANK",
    }}
    env.update(MODEL_ROOT=str(model), LORA_PATH=str(lora), OUTPUT_DIR=str(output), PYTHON_BIN=sys.executable)
    result = subprocess.run(["bash", str(checkout / "infer_batch_lora.sh")], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == exit_code, result.stdout + result.stderr
    observed = json.loads((output / "observed.json").read_text())
    assert observed["MODEL_ROOT"] == str(model)
    assert observed["DATA_ROOT"] == str(data)
    assert observed["METADATA_PATH"] == str(data / "metadata_6cases_480x832.csv")
    assert observed["EXPECTED_LORA_RANK"] == "64"
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        assert observed[variable] == "1"
    assert observed["DIFFSYNTH_SKIP_DOWNLOAD"] == "True"
    log = (output / "inference.log").read_text()
    assert "stdout evidence" in log and "stderr evidence" in log
