import os
from epspkit import io, preprocess, template, plotting
from epspkit.base import RecordingResult, PreprocessParams

base_path = r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\epsp-kit\epsp-kit\src\data"


all_files = [r'C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\epsp-kit\epsp-kit\src\data\\2024_05_27_0001.abf',r'C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\epsp-kit\epsp-kit\src\data\\2024_05_27_0002.abf']

# '2024_05_30_0002.abf',
# '2024_05_30_0004.abf',
# '2024_05_30_0006.abf',
# '2024_06_05_0000.abf',
# '2024_06_05_0001.abf']

# all_files = ['2025_05_22_0007.abf', '2025_05_22_0009.abf', '2025_05_22_0003.abf', '2025_05_22_0001.abf']

# all_files = [os.path.join(base_path, f) for f in all_files]


def main() -> None:
    data = io.load_bulk(
        all_files, 
        intensities=[25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600], id_values=["a","b"], 
        repnum=5
    )
    params = PreprocessParams(
        baseline_window=(0, 0.1),
        artifact_window=(0, 2.5),
        artifact="interp",
        smoothing="savgol",
        smoothing_params={
            "size": 7,
            "window_length": 15,
            "polyorder": 3,
            "cutoff": 2000.0,
            "order": 3,
        },
    )

    preprocessed = preprocess.preprocess(
        data,
        baseline_window=params.baseline_window,
        artifact_window=params.artifact_window,
        artifact=params.artifact,
        smoothing=params.smoothing,
        smoothing_params=params.smoothing_params,
    )

    recording_result = RecordingResult(preprocess_params=params)

    ## add R^2 threshold for fit_template?

    recording_result.add(
        "result_fv",
        template.fit_template(
            preprocessed,
            template_window=(1.0, 2.5),
            search_window=(0.5, 4.0),
            template_intensities=[200, 250, 300, 400, 500, 600],
        )
    )

    recording_result.add(
        "result_fepsp",
        template.fit_template(
            preprocessed,
            template_window=(3.0, 4.7),
            search_window=(2.5, 10.0),
            template_intensities=[200, 250, 300, 400, 500, 600],
            slope_transform=True
        )
    )

    plotting.plot_trace(
        preprocessed,
        recording_result=recording_result,
        id_value="a",
        features=["result_fv", "result_fepsp"],
        intensities=[150, 200],
        annotated=True,
    )

    plotting.plot_io_curve(
        recording_result, features=["result_fv", "result_fepsp"],
        intensities=[150, 200],
    )

if __name__ == "__main__":
    main()
