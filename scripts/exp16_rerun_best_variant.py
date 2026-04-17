import argparse
import json
import subprocess
import sys
from pathlib import Path


def latest_summary(prefix: str) -> Path:
    matches = sorted(Path("experiments/results").glob(f"{prefix}_*.json"))
    if not matches:
        raise FileNotFoundError(f"No summaries found for prefix {prefix}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-summary", default=None, help="path to batch summary json; defaults to latest exp16_batch_h64_tcn summary")
    parser.add_argument("--label", default="exp16_best_rerun", help="artifact prefix for rerun")
    args = parser.parse_args()

    summary_path = Path(args.from_summary) if args.from_summary else latest_summary("exp16_batch_h64_tcn")
    summary = json.loads(summary_path.read_text())
    variant = summary["best_variant"]

    cmd = [
        sys.executable,
        "scripts/exp16_batch_h64_tcn.py",
        "--variants",
        variant,
        "--label",
        args.label,
    ]
    subprocess.run(cmd, check=True)

    rerun_summary = latest_summary(args.label)
    print(json.dumps({"source_summary": str(summary_path), "variant": variant, "rerun_summary": str(rerun_summary)}, indent=2))


if __name__ == "__main__":
    main()
