import os
from epspkit import io, preprocess, template

base_path = r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\epsp-kit\epsp-kit\src\epsp-kit\data"


all_files = [r'C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\epsp-kit\epsp-kit\src\epspkit\data\\2024_05_27_0001.abf']

# '2024_05_30_0002.abf',
# '2024_05_30_0004.abf',
# '2024_05_30_0006.abf',
# '2024_06_05_0000.abf',
# '2024_06_05_0001.abf']

# all_files = ['2025_05_22_0007.abf', '2025_05_22_0009.abf', '2025_05_22_0003.abf', '2025_05_22_0001.abf']

# all_files = [os.path.join(base_path, f) for f in all_files]


def main() -> None:
    data = io.load_bulk(all_files, intensities=[25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600], id_values=["a", "b", "c", "d"], repnum=5)
    preprocessed = preprocess.preprocess(data, baseline_window=(0,0.1), artifact_window=(0,1.0))
    template_res = template.fit_template(preprocessed)


if __name__ == "__main__":
    main()
