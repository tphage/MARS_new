"""
make_paper_plots.py
===================

Publication-quality evaluation plots sourced from a single-run
`aggregate_results.json` (same file the `visualize_evaluation_single.ipynb`
notebook uses).

Three figures are produced (each as PNG + PDF at 300 dpi) plus two aggregated
CSV summaries written under `paper_plots/`:

    1. total_average_bar.{png,pdf}           +  total_average_data.csv
         Bar chart: mean of all 12 rubric scores per pipeline.
    2. output_average_grouped_bar.{png,pdf}  +  output_average_data.csv
         Grouped bars: per-subsystem mean (4 criteria) for each pipeline.
    3. radar_chart_12_metrics.{png,pdf}
         Radar chart: 12 rubric criteria on the axes, one polygon per pipeline.

USAGE
-----
    python scripts/make_paper_plots.py

The loader searches for `aggregate_results.json` in this order:
    1. <project>/results/evaluation/aggregate_results.json
    2. <project>/results_old/evaluation/aggregate_results.json
    3. newest match of <project>/{results,results_old}/evaluation_runs/**/run_*/aggregate_results.json

Point `AGGREGATE_PATH` at a specific file if you need to override this.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ───────────────────────────────── CONFIG ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "paper_plots"
RUBRIC_PATH = PROJECT_ROOT / "config" / "evaluation_rubric.yaml"

# Either set an explicit absolute/relative path or leave as None for auto-resolution.
AGGREGATE_PATH: str | None = None
SEARCH_ROOTS = [PROJECT_ROOT / "results", PROJECT_ROOT / "results_old"]

# Pipeline display order (left-to-right in bars, first-to-last in radar legend).
CONDITION_ORDER = [
    "evaluation",
    "ablation_3agent",
    "ablation_1agent_rag",
    "ablation_1agent_no_rag",
]

# Subsystem grouping of the 12 rubric dimensions. Keys are the prefixes used in
# `evaluation_rubric.yaml` and `aggregate_results.json`; values are the axis
# labels that appear on the grouped bar chart.
SUBSYSTEMS = {
    "system1": "System 1\nProperty Specification",
    "system2": "System 2\nCandidate Selection",
    "system3": "System 3\nManufacturing Process",
}

SCORE_RANGE = (1.0, 5.0)

# Green monochromatic palette (dark -> light), matching the BeamPERL postprocessing style.
# Hatch patterns differentiate bars in B&W print.
COLORS = ["#1B5E20", "#2E7D32", "#66BB6A", "#A5D6A7"]
HATCHES = ["", "//", "--", "\\\\"]
LEGEND_TITLE = "Pipeline"

# Colorblind-safe palette (Okabe-Ito) plus distinct linestyles + markers for the radar chart,
# where green shades collapse in grayscale. Order matches CONDITION_ORDER.
RADAR_COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73"]
RADAR_LINESTYLES = ["-", "--", "-.", ":"]
RADAR_MARKERS = ["o", "s", "^", "D"]

DPI = 300

BAR_FIGSIZE = (5.33, 4.0)
GROUPED_FIGSIZE = (8.0, 4.8)
RADAR_FIGSIZE = (9.0, 9.0)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
    }
)


# ───────────────────────────────── LOADERS ──
def resolve_aggregate_path(explicit: str | None) -> Path:
    """Find the aggregate_results.json to plot; mirrors the notebook's search order."""
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    for root in SEARCH_ROOTS:
        direct = root / "evaluation" / "aggregate_results.json"
        if direct.is_file():
            return direct

    candidates: list[Path] = []
    for root in SEARCH_ROOTS:
        runs_dir = root / "evaluation_runs"
        if runs_dir.is_dir():
            candidates.extend(runs_dir.glob("**/run_*/aggregate_results.json"))
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0]

    searched = ", ".join(str(r) for r in SEARCH_ROOTS)
    raise FileNotFoundError(
        f"No aggregate_results.json found under: {searched}. Run scripts/run_evaluation.py first."
    )


def load_rubric(rubric_path: Path) -> dict:
    with open(rubric_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dimension_keys_in_order(rubric: dict) -> list[str]:
    """Canonical list of the 12 dimension keys, ordered as they appear in the rubric."""
    return list(rubric["dimensions"].keys())


def short_label(rubric: dict, dim_key: str) -> str:
    return rubric["dimensions"][dim_key].get("short_name", dim_key)


def criterion_label(rubric: dict, dim_key: str) -> str:
    return rubric["dimensions"][dim_key].get("criterion_label", dim_key)


def build_score_frame(
    aggregate: dict,
    rubric: dict,
    condition_order: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build a DataFrame: rows = pipelines (in given order if present), cols = 12 dim keys.

    Also returns the list of pipeline labels used as the DataFrame index, so
    plotting code can round-trip them.
    """
    scores = aggregate.get("aggregate_scores") or {}
    if not scores:
        raise ValueError("aggregate_results.json has no 'aggregate_scores'.")

    dim_keys = dimension_keys_in_order(rubric)
    present = [ck for ck in condition_order if ck in scores] or list(scores.keys())

    rows: dict[str, list[float]] = {}
    for ck in present:
        block = scores.get(ck) or {}
        label = block.get("label", ck)
        row = []
        for dk in dim_keys:
            v = block.get(dk)
            if not isinstance(v, (int, float)):
                raise ValueError(f"Missing score for condition '{ck}', dimension '{dk}'.")
            row.append(float(v))
        rows[label] = row

    df = pd.DataFrame.from_dict(rows, orient="index", columns=dim_keys)
    df.index.name = "pipeline"
    return df, list(rows.keys())


# ───────────────────────────────── AGGREGATION ──
def compute_total_average(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pipeline mean across all 12 rubric scores."""
    return (
        df.mean(axis=1)
        .rename("total_average")
        .reset_index()
        .rename(columns={"index": "pipeline"})
    )


def compute_output_averages(
    df: pd.DataFrame,
    rubric: dict,
    subsystems: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """For each pipeline, mean of the criteria in each subsystem.

    Returns the DataFrame and the subsystem display labels in order.
    """
    dim_keys = dimension_keys_in_order(rubric)
    labels: list[str] = []
    data: dict[str, pd.Series] = {}
    for prefix, display in subsystems.items():
        cols = [k for k in dim_keys if k.startswith(prefix + "_")]
        if not cols:
            continue
        data[display] = df[cols].mean(axis=1)
        labels.append(display)
    result = pd.DataFrame(data)
    result.index.name = "pipeline"
    return result.reset_index(), labels


# ───────────────────────────────── PLOTS ──
def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI)
        print(f"[saved] {path}")
    plt.close(fig)


def plot_total_average(total_df: pd.DataFrame, out_dir: Path) -> None:
    pipelines = total_df["pipeline"].tolist()
    values = total_df["total_average"].to_numpy()
    x = np.arange(len(pipelines))

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    for xi, v, color, hatch in zip(x, values, COLORS, HATCHES):
        ax.bar(
            xi,
            v,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            hatch=hatch,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, rotation=15, ha="right")
    ax.set_ylabel("Average score (all 12 criteria)")
    ax.set_ylim(0, SCORE_RANGE[1] + 0.3)

    for xi, v in zip(x, values):
        ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    _save(fig, out_dir, "total_average_bar")


def plot_output_averages(
    out_df: pd.DataFrame,
    subsystem_labels: list[str],
    out_dir: Path,
) -> None:
    pipelines = out_df["pipeline"].tolist()
    data = out_df[subsystem_labels].to_numpy()  # [n_pipelines, n_subsystems]
    n_outputs = len(subsystem_labels)
    n_pipelines = len(pipelines)

    x = np.arange(n_outputs)
    width = 0.8 / n_pipelines

    fig, ax = plt.subplots(figsize=GROUPED_FIGSIZE)
    for i, pipeline in enumerate(pipelines):
        pos = x - 0.4 + width / 2 + i * width
        ax.bar(
            pos,
            data[i],
            width,
            color=COLORS[i % len(COLORS)],
            edgecolor="black",
            linewidth=1.0,
            hatch=HATCHES[i % len(HATCHES)],
            label=pipeline,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(subsystem_labels)
    ax.set_ylabel("Average score (mean of 4 criteria)")
    ax.set_ylim(0, SCORE_RANGE[1] + 0.3)
    ax.legend(title=LEGEND_TITLE, loc="upper right", frameon=True)

    _save(fig, out_dir, "output_average_grouped_bar")


def plot_radar_chart(
    df: pd.DataFrame,
    rubric: dict,
    out_dir: Path,
) -> None:
    dim_keys = dimension_keys_in_order(rubric)
    labels = [short_label(rubric, k) for k in dim_keys]
    n = len(dim_keys)
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    r_min, r_max = SCORE_RANGE

    fig, ax = plt.subplots(figsize=RADAR_FIGSIZE, subplot_kw=dict(projection="polar"))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for i, pipeline in enumerate(df.index):
        values = df.loc[pipeline, dim_keys].to_numpy(dtype=float)
        v_closed = np.concatenate([values, values[:1]])
        a_closed = np.concatenate([angles, angles[:1]])
        color = RADAR_COLORS[i % len(RADAR_COLORS)]
        linestyle = RADAR_LINESTYLES[i % len(RADAR_LINESTYLES)]
        marker = RADAR_MARKERS[i % len(RADAR_MARKERS)]
        ax.plot(
            a_closed,
            v_closed,
            linestyle=linestyle,
            linewidth=1.8,
            marker=marker,
            markersize=5,
            color=color,
            label=pipeline,
        )
        ax.fill(a_closed, v_closed, alpha=0.06, color=color)

    r_ticks = np.linspace(r_min, r_max, 5)
    ax.set_ylim(r_min, r_max)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels(
        [f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}" for v in r_ticks],
        fontsize=8,
    )
    ax.set_rlabel_position(180.0 / n)

    ax.set_xticks(angles)
    ax.set_xticklabels([""] * n)

    label_pad = (r_max - r_min) * 0.10
    for angle, label in zip(angles, labels):
        # Align each axis label based on its Cartesian direction so the text
        # sits naturally outside the circle.
        math_angle = np.pi / 2 - angle
        dx = np.cos(math_angle)
        dy = np.sin(math_angle)
        ha = "center" if abs(dx) < 0.1 else ("left" if dx > 0 else "right")
        va = "center" if abs(dy) < 0.1 else ("bottom" if dy > 0 else "top")
        ax.text(angle, r_max + label_pad, label, ha=ha, va=va, fontsize=9)

    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    ax.legend(
        title=LEGEND_TITLE,
        loc="upper right",
        bbox_to_anchor=(1.30, 1.10),
        frameon=True,
    )

    _save(fig, out_dir, "radar_chart_12_metrics")


# ───────────────────────────────── MAIN ──
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paper figures from aggregate_results.json",
    )
    parser.add_argument(
        "--aggregate",
        "-a",
        default=None,
        help="Path to aggregate_results.json (default: auto-detect under results/)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    aggregate_path = resolve_aggregate_path(args.aggregate or AGGREGATE_PATH)
    print(f"[info] Using {aggregate_path}")

    with open(aggregate_path, encoding="utf-8") as f:
        aggregate = json.load(f)
    rubric = load_rubric(RUBRIC_PATH)

    df, pipelines = build_score_frame(aggregate, rubric, CONDITION_ORDER)
    print(f"[info] Pipelines: {pipelines}")

    total_df = compute_total_average(df)
    total_csv = OUTPUT_DIR / "total_average_data.csv"
    total_df.to_csv(total_csv, index=False)
    print(f"[saved] {total_csv}")

    outputs_df, subsystem_labels = compute_output_averages(df, rubric, SUBSYSTEMS)
    outputs_csv = OUTPUT_DIR / "output_average_data.csv"
    outputs_df.to_csv(outputs_csv, index=False)
    print(f"[saved] {outputs_csv}")

    plot_total_average(total_df, OUTPUT_DIR)
    plot_output_averages(outputs_df, subsystem_labels, OUTPUT_DIR)
    plot_radar_chart(df, rubric, OUTPUT_DIR)

    print(f"[done] All figures and CSVs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
