import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from evoked.base import RecordingResult

def event_classifier(recording_result: RecordingResult, features: list[str], r2_cutoff: float):
        return None

def pop_spike_threshold(
    events: np.ndarray,
    slope: np.ndarray,
    x_label: str = "fEPSP Scale",
):
    """For population spike
    Arguments:
        events: np.ndarray an array of bools
        slope: np.ndarray an array of floats
    """
    events = np.asarray(events, dtype=int)
    slope = np.asarray(slope, dtype=float).reshape(-1,1) # must be 2D

    ps_model = LogisticRegression()
    ps_model.fit(slope, events)

    ps_thresh = -ps_model.intercept_[0] / ps_model.coef_[0][0] # slope where prob of ps exceeds 0.5

    slope_smooth = np.linspace(slope.min(), slope.max(), 100).reshape(-1, 1)

    ps_probs = ps_model.predict_proba(slope_smooth)[:, 1]

    plt.scatter(slope, events, label='Actual Data')
    plt.plot(slope_smooth, ps_probs, label='Logistic Regression', color='blue')
    plt.xlabel(x_label)
    plt.ylabel("Prob(Population Spike)")
    plt.legend()
    plt.show()
    return ps_thresh

if __name__ == "__main__":
    e = np.array([0,1,0,0,1,0,1,1])
    x = np.array([1.2,3.3, 2.5, 0.9,1.45,1.7,2.8,3.1])

    ps = pop_spike_threshold(e, x)
    print(ps)

def plot_single_intensity_pca(prep_df: pd.DataFrame, target_intensity: int = 600):
    """Isolates a single intensity step to uncover unsupervised shape and feature

    clustering across slices.
    """
    # 1. Filter strictly for the chosen intensity step
    sub_df = prep_df[prep_df["intensity"] == target_intensity]

    if sub_df.empty:
        raise ValueError(
            f"No data found in the dataframe for intensity {target_intensity}"
        )

    # 2. Pivot long format to wide matrix (rows = slices, columns = time points)
    pivot_df = sub_df.pivot(index="id", columns="time", values="voltage")
    X = pivot_df.values
    slice_ids = pivot_df.index.tolist()
    time_points = pivot_df.columns.to_numpy()

    # Center the matrix columns (time points) to focus entirely on shape variance
    X_centered = X - np.mean(X, axis=0)

    # 3. Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_centered)

    # 4. Set up a multi-panel figure to see both the clusters and the features driving them
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    # Panel 1: The Slice Cluster Map
    ax1.axhline(0, color="grey", linestyle="--", alpha=0.4)
    ax1.axvline(0, color="grey", linestyle="--", alpha=0.4)

    # Scatter plot of individual slices
    ax1.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        color="skyblue",
        edgecolors="black",
        s=80,
        alpha=0.9,
        zorder=3,
    )

    # Annotate dots with their unique file prefix IDs so you can find outliers instantly
    for i, slice_id in enumerate(slice_ids):
        ax1.annotate(
            f" {slice_id}",
            (X_pca[i, 0], X_pca[i, 1]),
            fontsize=8,
            alpha=0.8,
            zorder=4,
        )

    ax1.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance Explained)"
    )
    ax1.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance Explained)"
    )
    ax1.set_title(
        f"Slice Feature Clustering at {target_intensity} µA",
        fontweight="bold",
    )
    ax1.grid(alpha=0.2, zorder=1)

    # Panel 2: The Driving Eigen-Waveforms (What do PC1 and PC2 physically mean?)
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3)

    # Multiply components by a scalar to match voltage scale for overlay visualization
    ax2.plot(
        time_points,
        pca.components_[0],
        label="PC1 (Shape Modifier 1)",
        color="firebrick",
        linewidth=2,
    )
    ax2.plot(
        time_points,
        pca.components_[1],
        label="PC2 (Shape Modifier 2)",
        color="darkorange",
        linewidth=2,
    )

    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Component Weight (Arbitrary Units)")
    ax2.set_title("Underlying Morphological Axes", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"ca1_feature_clustering_{target_intensity}ua.png", dpi=150)
    plt.close()

    print(
        f"PCA complete for {target_intensity}uA. Plot saved to ca1_feature_clustering_{target_intensity}ua.png"
    )

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
