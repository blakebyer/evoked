import os
from epspkit.io import load_bulk
from epspkit.preprocess import preprocess
from epspkit.template import match_feature
from epspkit.plotting import plot_trace, plot_io_curve, plot_fit
from epspkit.base import RecordingResult, PreprocessParams
from epspkit.base import RecordingResult, PreprocessParams

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


base_path = r"C:\Users\bbyer\OneDrive\Documents\UniversityofKentucky\BachstetterLab\epsp-kit\epsp-kit\src\data"

# 29 mice, 37 ca1 slices, 34 dg slices
# 9 ca1 template slices, 6 dg template slices. So 28 test slices each

all_files_ca1 = ['2025_05_22_0006.abf', '2025_05_22_0004.abf', '2025_05_22_0000.abf', '2025_03_07_0004.abf', '2025_03_07_0008.abf', '2025_03_07_0006.abf', '2025_03_07_0002.abf', '2025_03_07_0000.abf', '2025_03_06_0002.abf', '2025_03_06_0004.abf', '2025_03_06_0010.abf', '2025_03_06_0000.abf', '2025_03_06_0006.abf', '2025_03_06_0008.abf', '2025_03_06_0011.abf', '2025_03_05_0006.abf', '2025_03_05_0004.abf', '2025_03_05_0008.abf', '2025_03_05_0002.abf', '2025_03_05_0000.abf', '2025_03_04_0007.abf', '2025_03_04_0000.abf', '2025_03_04_0002.abf', '2025_03_04_0004.abf', '2025_03_04_0006.abf', '2025_03_04_0010.abf', '2025_03_04_0012.abf', '2025_03_03_0005.abf', '2025_03_03_0000.abf', '2025_03_03_0002.abf', '2025_03_03_0003.abf', '2025_03_03_0009.abf', '2025_03_03_0007.abf', '2025_03_02_0006.abf', '2025_03_02_0009.abf', '2025_03_02_0002.abf', '2025_03_02_0000.abf']

mouse_id_ca1 = ['ephys-1-144-04', 'ephys-1-144-03', 'ephys-1-144-01', 'ephys-1-85-07', 'ephys-1-85-09', 'ephys-1-85-08', 'ephys-1-85-06', 'ephys-1-85-06', 'ephys-1-85-01', 'ephys-1-85-02', 'ephys-1-85-03', 'ephys-1-85-01', 'ephys-1-85-02', 'ephys-1-85-03', 'ephys-1-85-04', 'ephys-1-90-09', 'ephys-1-90-08', 'ephys-1-90-10', 'ephys-1-90-07', 'ephys-1-90-06', 'ephys-1-90-03', 'ephys-1-90-01', 'ephys-1-90-01', 'ephys-1-90-02', 'ephys-1-90-02', 'ephys-1-90-04', 'ephys-1-90-05', 'ephys-1-75-08', 'ephys-1-75-06', 'ephys-1-75-06', 'ephys-1-75-07', 'ephys-1-75-10', 'ephys-1-75-09', 'ephys-1-75-03', 'ephys-1-75-05', 'ephys-1-75-01', 'ephys-1-75-01']

all_files_dg = ['2025_05_22_0007.abf', '2025_05_22_0001.abf', '2025_05_22_0005.abf', '2025_03_07_0009.abf', '2025_03_07_0007.abf', '2025_03_07_0005.abf', '2025_03_07_0003.abf', '2025_03_06_0003.abf', '2025_03_06_0001.abf', '2025_03_06_0005.abf', '2025_03_06_0012.abf', '2025_03_06_0007.abf', '2025_03_06_0009.abf', '2025_03_05_0005.abf', '2025_03_05_0007.abf', '2025_03_05_0009.abf', '2025_03_05_0003.abf', '2025_03_05_0001.abf', '2025_03_04_0001.abf', '2025_03_04_0003.abf', '2025_03_04_0005.abf', '2025_03_04_0008.abf', '2025_03_04_0009.abf', '2025_03_04_0011.abf', '2025_03_04_0013.abf', '2025_03_03_0001.abf', '2025_03_03_0004.abf', '2025_03_03_0006.abf', '2025_03_03_0008.abf', '2025_03_03_0010.abf', '2025_03_02_0003.abf', '2025_03_02_0010.abf', '2025_03_02_0001.abf', '2025_03_02_0007.abf']

mouse_id_dg = ['ephys-1-144-04', 'ephys-1-144-01', 'ephys-1-144-03', 'ephys-1-85-09', 'ephys-1-85-08', 'ephys-1-85-07', 'ephys-1-85-06', 'ephys-1-85-01', 'ephys-1-85-01', 'ephys-1-85-02', 'ephys-1-85-04', 'ephys-1-85-02', 'ephys-1-85-03', 'ephys-1-90-08', 'ephys-1-90-09', 'ephys-1-90-10', 'ephys-1-90-07', 'ephys-1-90-06', 'ephys-1-90-01', 'ephys-1-90-01', 'ephys-1-90-02', 'ephys-1-90-03', 'ephys-1-90-03', 'ephys-1-90-04', 'ephys-1-90-05', 'ephys-1-75-06', 'ephys-1-75-07', 'ephys-1-75-08', 'ephys-1-75-09', 'ephys-1-75-10', 'ephys-1-75-01', 'ephys-1-75-05', 'ephys-1-75-01', 'ephys-1-75-03']

template_files_ca1 = ['2025_03_04_0002.abf','2025_03_05_0004.abf', '2025_03_06_0010.abf', '2025_03_02_0000.abf', '2025_03_07_0002.abf', '2025_03_06_0004.abf', '2025_03_05_0000.abf', '2025_03_04_0012.abf', '2025_03_03_0007.abf']
template_files_dg = ['2025_05_22_0005.abf', '2025_03_06_0001.abf','2025_03_05_0007.abf','2025_03_04_0005.abf','2025_03_04_0003.abf','2025_03_05_0009.abf']

# 1. Extract the file prefix (e.g., '2025_03_06_0002') to use as the unique ID
ca1_map = {f: f.split('.')[0] for f in all_files_ca1}
dg_map = {f: f.split('.')[0] for f in all_files_dg}

# 2. Filter out the template files to get your test files
ca1_test_files = [f for f in all_files_ca1 if f not in template_files_ca1]
dg_test_files = [f for f in all_files_dg if f not in template_files_dg]

# 3. Look up the now-unique file prefix IDs
ca1_test_ids = [ca1_map[f] for f in ca1_test_files]
dg_test_ids = [dg_map[f] for f in dg_test_files]
ca1_train_ids = [ca1_map[f] for f in template_files_ca1]
dg_train_ids = [dg_map[f] for f in template_files_dg]

# 4. Join paths
template_files_ca1_full = [os.path.join(base_path, f) for f in template_files_ca1]
template_files_dg_full = [os.path.join(base_path, f) for f in template_files_dg]
ca1_test_files_full = [os.path.join(base_path, f) for f in ca1_test_files]
dg_test_files_full = [os.path.join(base_path, f) for f in dg_test_files]

def main() -> None:
    intensities_list = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600]
    template_ints = [200, 250, 300, 400, 500, 600]
    
    params = PreprocessParams(
        baseline_window=(0, 0.1),
        artifact_window=(0, 2.5),
        artifact="zero",
        smoothing="savgol",
        smoothing_params={
            "size": 7,
            "window_length": 15,
            "polyorder": 3,
            "cutoff": 2000.0,
            "order": 3,
        },
    )

    # -------------------------------------------------------------
    # CA1 Pipeline
    # -------------------------------------------------------------
    print("Processing CA1...")
    ca1_raw_train = load_bulk(template_files_ca1_full, intensities=intensities_list, id_values=ca1_train_ids, repnum=3)
    ca1_raw_test = load_bulk(ca1_test_files_full, intensities=intensities_list, id_values=ca1_test_ids, repnum=3)

    ca1_prep_train = preprocess(ca1_raw_train, params.baseline_window, params.artifact_window, params.artifact, params.smoothing, params.smoothing_params)
    ca1_prep_test = preprocess(ca1_raw_test, params.baseline_window, params.artifact_window, params.artifact, params.smoothing, params.smoothing_params)

    #plot_single_intensity_pca(ca1_prep_test, target_intensity=600)

    ca1_results = RecordingResult(preprocess_params=params)
    
    ca1_results.add("result_fv", match_feature(
        train_df=ca1_prep_train, test_df=ca1_prep_test, 
        template_window=(1.0, 2.5), search_window=(0.5, 4.0), 
        template_intensities=template_ints, slope_transform=False
    ))
    
    ca1_results.add("result_fepsp", match_feature(
        train_df=ca1_prep_train, test_df=ca1_prep_test, 
        template_window=(3.0, 4.7), search_window=(2.5, 10.0), 
        template_intensities=template_ints, slope_transform=True
    ))

    # # -------------------------------------------------------------
    # # DG Pipeline
    # # -------------------------------------------------------------
    # print("Processing DG...")
    # dg_raw_train = load_bulk(template_files_dg_full, intensities=intensities_list, id_values=dg_train_ids, repnum=3)
    # dg_raw_test = load_bulk(dg_test_files_full, intensities=intensities_list, id_values=dg_test_ids, repnum=3)

    # dg_prep_train = preprocess(dg_raw_train, params.baseline_window, (0,2.25), params.artifact, params.smoothing, params.smoothing_params)
    # dg_prep_test = preprocess(dg_raw_test, params.baseline_window, (0,2.25), params.artifact, params.smoothing, params.smoothing_params)

    # dg_results = RecordingResult(preprocess_params=params)
    
    # dg_results.add("result_fv", match_feature(
    #     train_df=dg_prep_train, test_df=dg_prep_test, 
    #     template_window=(1.0, 2.5), search_window=(0.5, 4.0), 
    #     template_intensities=template_ints, slope_transform=False
    # ))
    
    # dg_results.add("result_fepsp", match_feature(
    #     train_df=dg_prep_train, test_df=dg_prep_test, 
    #     template_window=(3.0, 4.7), search_window=(2.5, 10.0), 
    #     template_intensities=template_ints, slope_transform=True
    # ))

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    print("Plotting CA1 Responses...")
    plot_trace(ca1_prep_test, recording_result=ca1_results, id_value=ca1_test_ids[0], features=["result_fv", "result_fepsp"], intensities=[200, 250, 300, 400, 500, 600], annotated=True)
    plot_io_curve(ca1_results, features=["result_fv", "result_fepsp"], intensities=[200, 250, 300, 400, 500, 600])
    plot_fit(ca1_prep_test, ca1_results, features=["result_fv", "result_fepsp"], intensity=600, id_value=ca1_test_ids[0])

    # print("Plotting DG Responses...")
    # plot_trace(dg_prep_test, recording_result=dg_results, id_value=dg_test_ids[0], features=["result_fv", "result_fepsp"], intensities=[200, 250, 300, 400, 500, 600], annotated=True)
    # plot_io_curve(dg_results, features=["result_fv", "result_fepsp"], intensities=[200, 250, 300, 400, 500, 600])

if __name__ == "__main__":
    main()
