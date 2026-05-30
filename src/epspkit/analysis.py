import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from epspkit.base import RecordingResult

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
        f"Slice Feature Clustering at {target_intensity} $\mu$A",
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
