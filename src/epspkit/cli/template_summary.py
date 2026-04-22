from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "fiber_volley": "Fiber volley",
    "epsp": "EPSP",
    "pop_spike": "Population spike",
}

R2_COLOR = "#DB4C77"
SCALE_COLOR = "#10559A"


def _sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = len(values)
    if n < 2:
        return np.nan
    return float(values.std(ddof=1) / math.sqrt(n))


def _stem_without_results_suffix(path: Path) -> str:
    stem = path.stem
    return stem[:-8] if stem.endswith("_results") else stem


def find_result_files(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.exists() and path.is_dir():
        return sorted(path.glob("*_results.xlsx"))
    return sorted(Path().glob(input_path))


def load_feature_tables(result_files: list[Path]) -> dict[str, pd.DataFrame]:
    feature_tables: dict[str, list[pd.DataFrame]] = {}

    for file_path in result_files:
        xls = pd.ExcelFile(file_path)
        result_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("result_")]

        for sheet in result_sheets:
            feature_name = sheet.removeprefix("result_")
            df = pd.read_excel(file_path, sheet_name=sheet)

            required_cols = {"stim_intensity", "template_r2", "template_scale"}
            if not required_cols.issubset(df.columns):
                continue

            df = df.copy()
            df["file_id"] = _stem_without_results_suffix(file_path)
            df["feature"] = feature_name
            feature_tables.setdefault(feature_name, []).append(df)

    return {
        feature_name: pd.concat(tables, ignore_index=True)
        for feature_name, tables in feature_tables.items()
        if tables
    }


def summarize_feature(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("stim_intensity", dropna=False)
        .agg(
            r2_mean=("template_r2", "mean"),
            r2_sem=("template_r2", _sem),
            r2_n=("template_r2", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
            scale_mean=("template_scale", "mean"),
            scale_sem=("template_scale", _sem),
            scale_n=("template_scale", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
        )
        .reset_index()
        .sort_values("stim_intensity")
    )
    return summary


def plot_feature_summary(
    feature_name: str,
    summary: pd.DataFrame,
    output_dir: Path,
    show: bool = False,
) -> Path:
    label = FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)
    x = summary["stim_intensity"].to_numpy()

    axes[0].errorbar(
        x,
        summary["r2_mean"].to_numpy(),
        yerr=summary["r2_sem"].to_numpy(),
        fmt="-o",
        color=R2_COLOR,
        linewidth=2,
        elinewidth=1.5,
        capsize=4,
    )
    axes[0].set_ylabel(r"Mean template $R^2$")
    axes[0].set_title(label)
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(
        x,
        summary["scale_mean"].to_numpy(),
        yerr=summary["scale_sem"].to_numpy(),
        fmt="-o",
        color=SCALE_COLOR,
        linewidth=2,
        elinewidth=1.5,
        capsize=4,
    )
    axes[1].set_xlabel("Stimulus intensity")
    axes[1].set_ylabel("Mean template scale")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / f"{feature_name}_template_summary.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool template_r2 and template_scale across *_results.xlsx files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="test2",
        help="Directory containing *_results.xlsx files, or a glob pattern. Default: test2",
    )
    parser.add_argument(
        "--output-dir",
        default="test2",
        help="Directory for output plots and summaries. Default: test2",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively.",
    )
    args = parser.parse_args()

    result_files = find_result_files(args.input)
    if not result_files:
        raise SystemExit("No *_results.xlsx files found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_tables = load_feature_tables(result_files)
    if not feature_tables:
        raise SystemExit(
            "No usable result_* sheets containing template_r2/template_scale were found."
        )

    written = []
    for feature_name, df in feature_tables.items():
        summary = summarize_feature(df)
        if summary.empty:
            continue

        summary.to_csv(output_dir / f"{feature_name}_template_summary.csv", index=False)
        written.append(plot_feature_summary(feature_name, summary, output_dir, show=args.show))

    if not written:
        raise SystemExit("No summary plots were written.")

    print("Wrote summary plots:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
