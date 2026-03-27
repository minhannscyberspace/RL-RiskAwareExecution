from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_eval_report(eval_dir: str | Path) -> dict[str, str]:
    base = Path(eval_dir)
    results_path = base / "results.csv"
    summary_path = base / "summary.csv"
    if not results_path.exists() or not summary_path.exists():
        raise FileNotFoundError("results.csv and summary.csv must exist in eval_dir")

    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path)

    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric_by_policy(results, "completion", plots_dir / "completion_by_policy.png", ylim=(0.0, 1.05))
    _plot_metric_by_policy(results, "is", plots_dir / "is_by_policy.png")
    _plot_metric_by_policy(results, "slippage_bps", plots_dir / "slippage_bps_by_policy.png")

    report_path = base / "report.md"
    _write_markdown_report(report_path, summary, plots_dir)
    html_path = base / "report.html"
    _write_html_report(html_path, summary, plots_dir)
    return {"report_md": str(report_path), "report_html": str(html_path), "plots_dir": str(plots_dir)}


def _plot_metric_by_policy(df: pd.DataFrame, metric: str, out_path: Path, ylim: tuple[float, float] | None = None) -> None:
    grouped = df.groupby("policy", as_index=False)[metric].mean().sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grouped["policy"], grouped[metric])
    ax.set_title(f"{metric} (mean by policy)")
    ax.set_xlabel("Policy")
    ax.set_ylabel(metric)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_markdown_report(report_path: Path, summary: pd.DataFrame, plots_dir: Path) -> None:
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## Policy Summary (from summary.csv)")
    lines.append("")
    lines.extend(_df_to_markdown_table(summary))
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for plot_name in ("completion_by_policy.png", "is_by_policy.png", "slippage_bps_by_policy.png"):
        rel = Path("plots") / plot_name
        lines.append(f"### {plot_name}")
        lines.append(f"![{plot_name}]({rel.as_posix()})")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_html_report(report_path: Path, summary: pd.DataFrame, plots_dir: Path) -> None:
    table_html = summary.to_html(index=False)
    plot_blocks = []
    for plot_name in ("completion_by_policy.png", "is_by_policy.png", "slippage_bps_by_policy.png"):
        rel = (Path("plots") / plot_name).as_posix()
        plot_blocks.append(f"<h3>{plot_name}</h3><img src=\"{rel}\" style=\"max-width: 900px;\"/>")
    html = (
        "<html><head><meta charset='utf-8'><title>Evaluation Report</title></head><body>"
        "<h1>Evaluation Report</h1>"
        "<h2>Policy Summary (from summary.csv)</h2>"
        f"{table_html}"
        "<h2>Plots</h2>"
        + "".join(plot_blocks)
        + "</body></html>"
    )
    report_path.write_text(html, encoding="utf-8")


def _df_to_markdown_table(df: pd.DataFrame) -> list[str]:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows: list[str] = [header, sep]
    for _, row in df.iterrows():
        vals: list[str] = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append("nan")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return rows
