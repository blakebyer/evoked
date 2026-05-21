import os
from epspkit import io, preprocess, template

base_path = r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\epsp-kit\epsp-kit\src\data"

all_files = ['2025_05_22_0007.abf', '2025_05_22_0009.abf', '2025_05_22_0003.abf', '2025_05_22_0001.abf']

all_files = [os.path.join(base_path, f) for f in all_files]


def main() -> None:
    data = io.load_bulk(all_files, intensities=[25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600], id_values=["a", "b", "c", "d"], repnum=3)
    print(type(data))
    print(data)


if __name__ == "__main__":
    main()
