from __future__ import annotations

import argparse

from rl_riskaware.reporting import build_eval_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown+plot report from eval artifact directory.")
    parser.add_argument("--eval-dir", type=str, required=True)
    args = parser.parse_args()

    out = build_eval_report(args.eval_dir)
    print(f"Report generated: {out['report_md']}")
    print(f"HTML report: {out['report_html']}")
    print(f"Plots dir: {out['plots_dir']}")


if __name__ == "__main__":
    main()
