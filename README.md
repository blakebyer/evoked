# evoked
evoked is a package for analyzing evoked local field potentials using template matching.

## What it does
- Load ABF/CSV recordings into tidy pandas DataFrames
- Preprocesses tidy data (baseline correction, stim artifact removal, averaging)
- Matches features (e.g., fiber volley, fEPSP slope, population spike)
- Renders common plots (e.g., IO curves)

## Installation
### 1. Install Poetry
```bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry
```
### 2. Install evoked
```bash
git clone https://github.com/blakebyer/evoked.git
cd evoked
poetry install
```

## Usage
### Python API ###
The Python API is recommended for heavy users.

Edit `main.py` and run:
```bash
poetry run python -m evoked.main
```
See `examples/analysis_2026_06_19.ipynb` for detailed instructions. 

### Web App ###
evoked comes with a basic streamlit web app, which can be run by:
```bash
streamlit run app.py
```

## Output
- Results are written to one Excel (.xlsx) file with `save_results_xlsx` or JSON file with `save_results_json`.
- Each workbook contains sheets: `Pipeline` containing pipeline configuration information and one sheet per feature with feature parameters and results.
