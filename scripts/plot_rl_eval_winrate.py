"""Plot recorded RL evaluation win rates with 95% confidence intervals.

The training loop writes one ``arms-report.json`` at each evaluation point.
This utility discovers those reports, uses the report's binomial standard
error, and writes a plot and a CSV summary.

Example:
    python scripts/plot_rl_eval_winrate.py \
        data/training/mono-red-rl-4000-20260919-224841
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def read_evaluations(run_dir: Path) -> list[dict[str, float | int]]:
    rows = []
    for report_path in sorted(run_dir.glob("iter-*/arms-report.json")):
        match = re.fullmatch(r"iter-(\d+)", report_path.parent.name)
        if match is None:
            continue
        report = json.loads(report_path.read_text())
        if len(report) != 1:
            raise ValueError(f"expected one arm in {report_path}")
        arm = next(iter(report.values()))
        required = ("games", "model_wins", "winrate", "se")
        missing = [key for key in required if key not in arm]
        if missing:
            raise ValueError(f"missing {missing} in {report_path}")
        margin = 1.96 * float(arm["se"])
        rows.append(
            {
                "iteration": int(match.group(1)),
                "games": int(arm["games"]),
                "model_wins": int(arm["model_wins"]),
                "winrate": float(arm["winrate"]),
                "se": float(arm["se"]),
                "ci95_low": max(0.0, float(arm["winrate"]) - margin),
                "ci95_high": min(1.0, float(arm["winrate"]) + margin),
                "report": str(report_path),
            }
        )
    if not rows:
        raise SystemExit(f"no arms-report.json files found under {run_dir}")
    return rows


def write_csv(rows: list[dict[str, float | int]], out_csv: Path) -> None:
    fields = [
        "iteration",
        "games",
        "model_wins",
        "winrate",
        "se",
        "ci95_low",
        "ci95_high",
        "report",
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, float | int]], out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["iteration"] for row in rows]
    y = [row["winrate"] for row in rows]
    lower = [row["winrate"] - row["ci95_low"] for row in rows]
    upper = [row["ci95_high"] - row["winrate"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x,
        y,
        yerr=[lower, upper],
        fmt="o-",
        color="#c2410c",
        ecolor="#7c2d12",
        elinewidth=1.5,
        capsize=4,
        markersize=6,
        linewidth=1.8,
    )
    ax.axhline(0.5, color="#6b7280", linestyle="--", linewidth=1, alpha=0.8)
    ax.set(
        title=title,
        xlabel="RL iteration",
        ylabel="Mirror-match win rate",
        xticks=x,
        ylim=(0, 1),
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.99,
        0.02,
        "error bars: 95% CI (win rate ± 1.96 SE)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="PNG path; defaults to run_dir/analysis/eval_winrate.png")
    parser.add_argument("--csv", type=Path, help="CSV path; defaults to the PNG path with .csv")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_png = args.out or run_dir / "analysis" / "eval_winrate.png"
    out_csv = args.csv or out_png.with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = read_evaluations(run_dir)
    write_csv(rows, out_csv)
    write_plot(rows, out_png, f"RL mirror-match evaluation\n{run_dir.name}")
    for row in rows:
        print(
            f"iter {row['iteration']:>2}: {row['model_wins']}/{row['games']} "
            f"= {row['winrate']:.1%} "
            f"(95% CI {row['ci95_low']:.1%}–{row['ci95_high']:.1%})"
        )
    print(f"plot: {out_png}")
    print(f"data: {out_csv}")


if __name__ == "__main__":
    main()
