from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fitting import run_combined_workflow


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run combined CPMG + DOSY + inversion recovery T1 titration fitting workflow."
    )
    parser.add_argument("input_json", type=Path, help="Path to workflow input JSON file")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output file for fit results JSON",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text())
    result = run_combined_workflow(payload)

    output = json.dumps(result, indent=args.indent)
    print(output)
    if args.output_json is not None:
        args.output_json.write_text(output + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
