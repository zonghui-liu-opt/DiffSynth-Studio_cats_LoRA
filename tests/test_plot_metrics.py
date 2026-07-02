import subprocess
import sys

from metrics_utils import load_metrics
from tests.gen_fake_metrics import write_fake_metrics


def test_plot_metrics_outputs_pngs_and_summary(tmp_path):
    metrics_path = tmp_path / "fake_metrics.jsonl"
    output_dir = tmp_path / "plots"
    write_fake_metrics(metrics_path, total_steps=300)

    result = subprocess.run(
        [
            sys.executable,
            "plot_metrics.py",
            "--metrics_path",
            str(metrics_path),
            "--output_dir",
            str(output_dir),
            "--warmup_steps",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_dir / "loss.png").is_file()
    assert (output_dir / "throughput.png").is_file()
    assert "total_steps: 300" in result.stdout
    assert "last_20pct_avg_loss:" in result.stdout
    assert "steady_tokens_per_sec:" in result.stdout
    assert "steady_videos_per_hour:" in result.stdout

    rows = load_metrics(metrics_path)
    assert len(rows) == 300
    assert rows[-1]["loss"] == 0.123
