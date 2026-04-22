from __future__ import annotations
import os

from epspkit.core.config import (
    FeatureConfig,
    IOConfig,
    PipelineConfig,
    SmoothingConfig,
    TransformConfig,
    VizConfig,
)
from epspkit.pipeline.api import run_pipeline

# clean_trace_rc = {
#     # Transparent figure
#     "figure.facecolor": "none",
#     "axes.facecolor": "none",
#     "savefig.facecolor": "none",
#     "savefig.transparent": True,

#     # Kill grid and axes entirely
#     "axes.grid": False,
#     "axes.axisbelow": False,

#     # Remove all spines
#     "axes.spines.top": False,
#     "axes.spines.right": False,
#     "axes.spines.bottom": False,
#     "axes.spines.left": False,

#     # Remove ticks and tick labels
#     "xtick.bottom": False,
#     "xtick.top": False,
#     "xtick.labelbottom": False,
#     "ytick.left": False,
#     "ytick.right": False,
#     "ytick.labelleft": False,

#     # Don’t show labels or titles
#     "axes.labelsize": 0,
#     "axes.titlesize": 0,

#     # Make the trace visually dominant
#     "lines.linewidth": 3.0,
#     "lines.solid_capstyle": "round",
# }
base_path = r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\epsp-kit\epsp-kit\src\data"

all_files = ['2025_03_03_0000.abf', '2025_03_02_0008.abf', '2025_03_07_0008.abf', '2025_03_06_0006.abf', '2025_03_03_0005.abf', '2025_03_04_0002.abf', '2025_03_07_0004.abf', '2025_03_04_0000.abf', '2025_03_04_0006.abf', '2025_03_06_0002.abf', '2025_03_05_0004.abf', '2025_03_07_0000.abf', '2025_03_05_0008.abf', '2025_03_06_0008.abf', '2025_03_05_0006.abf', '2025_03_06_0011.abf', '2025_03_03_0002.abf', '2025_05_22_0006.abf', '2025_03_04_0004.abf', '2025_03_07_0006.abf', '2025_03_06_0010.abf', '2025_05_22_0000.abf', '2025_03_02_0000.abf','2025_05_22_0004.abf', '2025_03_07_0010.abf', '2025_03_07_0002.abf', '2025_03_06_0000.abf', '2025_03_06_0004.abf', '2025_03_05_0002.abf', '2025_03_05_0000.abf', '2025_03_04_0012.abf', '2025_03_04_0010.abf', '2025_03_04_0007.abf', '2025_03_03_0009.abf', '2025_03_03_0007.abf', '2025_03_03_0003.abf', '2025_03_02_0009.abf', '2025_03_02_0002.abf', '2025_03_02_0006.abf']

template_files = ['2025_03_02_0008.abf','2025_03_04_0002.abf','2025_03_05_0004.abf', '2025_03_06_0010.abf', '2025_03_02_0000.abf', '2025_03_07_0002.abf', '2025_03_06_0004.abf', '2025_03_05_0000.abf', '2025_03_04_0012.abf', '2025_03_03_0007.abf']

test_files = ['2025_03_04_0006.abf']

# template_files = ['2025_03_02_0008.abf','2025_03_04_0002.abf', '2025_03_04_0000.abf', '2025_03_06_0002.abf']

#test_files = [f for f in all_files if f not in template_files]
template_files = [os.path.join(base_path, f) for f in template_files]
test_files = [os.path.join(base_path, f) for f in test_files]
all_files = [os.path.join(base_path, f) for f in all_files]


def main() -> None:
    pipeline_config = PipelineConfig(
        io=IOConfig(
            template_files=template_files,
            test_files=test_files,
            output_path="test4/",
            plot_pdf_path="test2/all_plots.pdf",
            metadata={
            },
            repnum=3,
            stim_intensities=[
                25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600
            ],
            write_results=False,
            write_plots=True,
            write_plot_pdf=False,
            plot_pdf_plots_per_page=6,
            render_plots=False,
        ),
        transforms=[
            TransformConfig(
                name="baseline_correction",
                params={"baseline_window_ms": (0.0, 0.1)},
            ),
            TransformConfig(
                name="template_subtract_stim_artifact",
                params={"window_ms":(0.0,2.0)}
            ),
            TransformConfig(name="average_sweeps"),
        ],
        features=[
            FeatureConfig(
                name="fiber_volley",
                params={
                    "method": "peak",
                    "window_ms": (1.5, 3.0),
                    "search_window_ms": (1.0, 3.2),
                    "template_stim_intensities": [200, 250, 300, 400, 500, 600],
                    "template_r2_threshold": 0.8,
                    "height":0
                },
            ),
            FeatureConfig(
                name="epsp",
                params={
                    "method": "peak",
                    "window_ms": (2.7, 4.0),
                    "search_window_ms": (2.5, 5.0),
                    "template_stim_intensities": [200, 250, 300, 400, 500, 600],
                    "fit_distance": 4,
                    "template_r2_threshold": 0.30,
                    "height":0
                },
            ),
            FeatureConfig(
                name="pop_spike",
                params={
                    "method": "peak",
                    "window_ms": (4.5, 6.5),
                    "search_window_ms": (3.5, 8.0),
                    "template_r2_threshold": 0.7,
                    "template_stim_intensities": [400,500,600],
                    "height":-3.0
                },
            ),
        ],
        plots=[
            #VizConfig(name="sweep", stim_intensities=[200, 250, 300, 400, 500, 600]),
            VizConfig(name="annotated", stim_intensities=[25, 50, 75, 100]),
            #VizConfig(name="template", stim_intensities=[25, 50, 75, 100]),
        ],
        global_smoothing=SmoothingConfig(
            method="savgol",
            window_size=21,
            polyorder=3,
        ),
    )
    run_pipeline(pipeline_config)


if __name__ == "__main__":
    main()
