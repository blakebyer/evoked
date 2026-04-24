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

all_files = ['2025_05_22_0007.abf', '2025_05_22_0009.abf', '2025_05_22_0003.abf', '2025_05_22_0001.abf', '2025_05_22_0005.abf', '2025_03_07_0009.abf', '2025_03_07_0007.abf', '2025_03_07_0005.abf', '2025_03_07_0003.abf', '2025_03_06_0014.abf', '2025_03_06_0012.abf', '2025_03_06_0009.abf', '2025_03_06_0007.abf', '2025_03_06_0005.abf', '2025_03_06_0003.abf', '2025_03_06_0001.abf', '2025_03_05_0005.abf', '2025_03_05_0003.abf', '2025_03_05_0009.abf', '2025_03_05_0007.abf', '2025_03_05_0001.abf', '2025_03_04_0001.abf', '2025_03_04_0013.abf', '2025_03_04_0011.abf', '2025_03_04_0009.abf', '2025_03_04_0008.abf', '2025_03_04_0005.abf', '2025_03_04_0003.abf', '2025_03_03_0010.abf', '2025_03_03_0008.abf', '2025_03_03_0006.abf', '2025_03_03_0004.abf', '2025_03_03_0001.abf', '2025_03_02_0003.abf', '2025_03_02_0001.abf', '2025_03_02_0010.abf', '2025_03_02_0005.abf', '2025_03_02_0007.abf']

# all_files = ['2025_03_03_0000.abf', '2025_03_02_0008.abf', '2025_03_07_0008.abf', '2025_03_06_0006.abf', '2025_03_03_0005.abf', '2025_03_04_0002.abf', '2025_03_07_0004.abf', '2025_03_04_0000.abf', '2025_03_04_0006.abf', '2025_03_06_0002.abf', '2025_03_05_0004.abf', '2025_03_07_0000.abf', '2025_03_05_0008.abf', '2025_03_06_0008.abf', '2025_03_05_0006.abf', '2025_03_06_0011.abf', '2025_03_03_0002.abf', '2025_05_22_0006.abf', '2025_03_04_0004.abf', '2025_03_07_0006.abf', '2025_03_06_0010.abf', '2025_05_22_0000.abf', '2025_03_02_0000.abf','2025_05_22_0004.abf', '2025_03_07_0010.abf', '2025_03_07_0002.abf', '2025_03_06_0000.abf', '2025_03_06_0004.abf', '2025_03_05_0002.abf', '2025_03_05_0000.abf', '2025_03_04_0012.abf', '2025_03_04_0010.abf', '2025_03_04_0007.abf', '2025_03_03_0009.abf', '2025_03_03_0007.abf', '2025_03_03_0003.abf', '2025_03_02_0009.abf', '2025_03_02_0002.abf', '2025_03_02_0006.abf']

# template_files = ['2025_03_02_0008.abf','2025_03_04_0002.abf','2025_03_05_0004.abf', '2025_03_06_0010.abf', '2025_03_02_0000.abf', '2025_03_07_0002.abf', '2025_03_06_0004.abf', '2025_03_05_0000.abf', '2025_03_04_0012.abf', '2025_03_03_0007.abf']

test_files = ['2025_03_07_0008.abf']

# template_files = ['2025_03_02_0008.abf','2025_03_04_0002.abf', '2025_03_04_0000.abf', '2025_03_06_0002.abf']

#test_files = [f for f in all_files if f not in template_files]
#template_files = [os.path.join(base_path, f) for f in template_files]
test_files = [os.path.join(base_path, f) for f in test_files]
all_files = [os.path.join(base_path, f) for f in all_files]


def main() -> None:
    pipeline_config = PipelineConfig(
        io=IOConfig(
            input_files=test_files,
            template_files=None,
            test_files=None,
            output_path="test5/",
            plot_pdf_path="test4/all_plots_dg.pdf",
            metadata={
            },
            repnum=3,
            stim_intensities=[
                25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600
            ],
            write_results=True,
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
                    "method": "derivative",
                    "window_ms": (1.5, 3.0),
                    "search_window_ms": (1.0, 3.2),
                    "template_stim_intensities": [200, 250, 300, 400, 500, 600],
                    "score_threshold": 0.8,
                    "height":0.5
                },
            ),
            FeatureConfig(
                name="epsp",
                params={
                    "method": "derivative",
                    "window_ms": (2.7, 4.0),
                    "search_window_ms": (2.5, 5.0),
                    "template_stim_intensities": [200, 250, 300, 400, 500, 600],
                    "fit_distance": 4,
                    "score_threshold": 0.30,
                },
            ),
            FeatureConfig(
                name="pop_spike",
                params={
                    "method": "derivative",
                    "window_ms": (4.5, 6.5),
                    "search_window_ms": (3.5, 8.0),
                    "score_threshold": 0.7,
                    "template_stim_intensities": [400,500,600],
                    "height":0
                },
            ),
        ],
        plots=[
            #VizConfig(name="sweep", stim_intensities=[200, 250, 300, 400, 500, 600]),
            VizConfig(name="annotated", stim_intensities=[600]),
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
