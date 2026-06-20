from __future__ import annotations
import numpy as np
import pandas as pd
from evoked.base import IntermediateResult, FitResult, FeatureResult, RecordingResult, window_to_indices
from pandera.typing import DataFrame
from evoked.template import center_signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def build_eigentemplate(
    intermediate: DataFrame[IntermediateResult],
    template_window: tuple[float, float],
    template_intensities: list[int],
    slope_transform: bool = False,
):
    """Builds a template and returns it alongside its slope_transform state."""
    template_data = intermediate[intermediate["intensity"].isin(template_intensities)]
    if template_data.empty:
        raise ValueError("No traces found for template_intensities.")

    template_snippets = []
    for _, group in template_data.groupby(["id", "intensity"], sort=False):
        time = group["time"].to_numpy()
        signal = group["voltage"].to_numpy()

        if slope_transform:
            signal = np.gradient(signal, time)

        template_start, template_stop = window_to_indices(time, template_window)
        template_snippet = signal[template_start:template_stop]
        norm = np.linalg.norm(template_snippet)
        if norm == 0:
            continue
        template_scaled = center_signal(template_snippet) / norm
        template_snippets.append(template_scaled)
    
    if not template_snippets:
        raise ValueError("No nonzero template snippets found.")

    pca = PCA(n_components=3) # fit PCA with 3 component eigenvectors
    pca.fit(template_snippets)
    print(pca.explained_variance_ratio_)

    eigenvectors = pca.components_

    return eigenvectors, slope_transform

def fit_eigentemplate(
    intermediate: DataFrame[IntermediateResult],
    template_window: tuple[float, float],
    search_window: tuple[float, float],
    template_package: tuple[np.ndarray, bool],
    r2_threshold: float,
    n_components: int = 2,
) -> FeatureResult:
    """Fits a pre-built eigentemplate package tuple: (eigenvectors, slope_transform)"""
    eigenvectors, slope_transform = template_package

    if eigenvectors.ndim != 2:
        raise ValueError("Eigentemplate array must be 2D.")
    
    if eigenvectors.shape[1] < 3:
        raise ValueError("Eigentemplate must contain at least 3 samples.")
    
    if n_components > eigenvectors.shape[0]:
        raise ValueError("n_components exceeds number of available eigentemplates.")

    basis = eigenvectors[:n_components]
    template_len = basis.shape[1]

    center_idx = int(template_len // 2)
    left = center_idx
    right = template_len - center_idx - 1

    results = []

    for (id_value, intensity), group in intermediate.groupby(["id", "intensity"], sort=False):
        time = group["time"].to_numpy()
        signal = group["voltage"].to_numpy()

        if slope_transform:
            signal = np.gradient(signal, time)

        start_idx, stop_idx = window_to_indices(time, search_window)
        first_center = start_idx + left
        last_center = stop_idx - right

        if last_center <= first_center:
            raise ValueError("Search window is too small for this eigentemplate.")

        best_score = -np.inf

        best_result = {
            "id": id_value,
            "intensity": intensity,
            "feature_time": np.nan,
            "proj_magnitude": np.nan,
            "energy": np.nan,
            "energy_arr": np.array([]),
            "betas": np.array([]),
            "r2": np.nan,
            "detected": False,
        }

        energy_list = []

        for center in range(first_center, last_center):
            snippet = signal[center - left : center + right + 1]

            # Match preprocessing used when building eigentemplates
            snippet_centered = center_signal(snippet)

            # Project snippet onto eigentemplate basis
            betas = np.dot(basis, snippet_centered)

            # Projection energy = squared amount of signal in eigentemplate space
            energy = float(np.sum(betas ** 2))
            projection_magnitude = float(np.sqrt(energy))
            energy_list.append(energy)

            

            # Reconstruct fitted snippet
            fit = np.dot(betas, basis)

            rss = float(np.sum((snippet_centered - fit) ** 2))
            tss = float(np.sum(snippet_centered ** 2))
            r2 = 1 - rss / tss if tss > 0 else np.nan

            if np.isnan(r2) or r2 <= best_score:
                continue

            best_score = r2

            feature_time = float(time[center] * 1000)
            detected = np.isfinite(r2) and r2 >= r2_threshold

            best_result = {
                "id": id_value,
                "intensity": intensity,
                "feature_time": feature_time,
                "proj_magnitude": projection_magnitude,
                "energy": energy,
                "energy_arr": np.array([]),
                "betas": betas,
                "r2": r2,
                "detected": detected,
            }

        best_result["energy_arr"] = np.asarray(energy_list, dtype=float)
        results.append(best_result)

    return FeatureResult(
        search_window=search_window,
        template_window=template_window,
        slope_transform=slope_transform,
        template=eigenvectors,
        result=pd.DataFrame(results),
    )

def eigenmatch_feature(
    train_df: DataFrame[IntermediateResult],
    test_df: DataFrame[IntermediateResult],
    template_window: tuple[float, float],
    search_window: tuple[float, float],
    template_intensities: list[int],
    r2_threshold: float,
    slope_transform: bool = False,
) -> FeatureResult:
    """Builds a template from training data and fits it directly onto testing data."""
    template_package = build_eigentemplate(
        intermediate=train_df,
        template_window=template_window,
        template_intensities=template_intensities,
        slope_transform=slope_transform
    )
    return fit_eigentemplate(
        intermediate=test_df,
        template_window=template_window,
        search_window=search_window,
        template_package=template_package,
        r2_threshold=r2_threshold,
    )
        
