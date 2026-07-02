import argparse
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_fake_metrics(output_path, total_steps=300):
    from metrics_utils import MetricsWriter

    writer = MetricsWriter(output_path)
    for step in range(1, total_steps + 1):
        warmup = step <= 5
        step_time = 4.0 if warmup else 1.5 + 0.05 * math.sin(step / 9)
        loss = 1.2 * math.exp(-step / 180) + 0.05 * math.sin(step / 7)
        writer.write(
            {
                "step": step,
                "epoch": step // 60,
                "loss": loss,
                "step_time_sec": step_time,
                "tokens_per_sample": 12090,
                "samples_per_step": 4,
                "lr": 1e-4,
            }
        )
    writer.write(
        {
            "step": total_steps,
            "epoch": total_steps // 60,
            "loss": 0.123,
            "step_time_sec": 1.25,
            "tokens_per_sample": 12090,
            "samples_per_step": 4,
            "lr": 1e-4,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total_steps", type=int, default=300)
    args = parser.parse_args()
    write_fake_metrics(args.output, args.total_steps)


if __name__ == "__main__":
    main()
