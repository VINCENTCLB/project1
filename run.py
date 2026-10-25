"""Reproduce the competition submission end to end.

Loads the BRFSS data, selects the preprocessing method, ridge lambda, class
weight and decision threshold on a development split, evaluates the selected
configuration once on an untouched holdout split, then refits on all labelled
rows and writes a submission CSV.

Usage:
    python run.py
    python run.py --data-path path/to/dataset --output-dir path/to/results

The data directory must contain x_train.csv, y_train.csv and x_test.csv under
their original names, as downloaded from the competition platform.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from medical_predictions import run_pipeline

DEFAULT_DATA_PATH = ROOT / "data" / "dataset"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "dataset" / "integrated_results"


def main():
    """Parse arguments and run the full pipeline."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Directory holding x_train.csv, y_train.csv and x_test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics.json, validation_f1.png and submission.csv.",
    )
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        subsample=False,
        degree=1,
        lr=0.01,
        submission=True,
        preprocessing="compare",
        plots=True,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
