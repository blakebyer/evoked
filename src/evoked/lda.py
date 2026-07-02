"""Linear discriminant analysis"""
from __future__ import annotations
import numpy as np
import polars as pl
from evoked.base import IntermediateResult, FeatureResult, window_to_indices
from pandera.typing.polars import DataFrame
from sklearn.covariance import OAS # Oracle Approximating Shrinkage
from evoked.ols import center_signal

def estimate_noise_covariance(noise_snippets: list[np.ndarray]) -> np.ndarray:
    """Estimate diagonal noise covariance matrix"""
    X = np.vstack([center_signal(x) for x in noise_snippets])
    variances = np.var(X, axis=0)
    variances[variances <= 1e-20] = 1e-20 
    return np.diag(variances)

def estimate_score(snippet: np.ndarray, template: np.ndarray, covariance_matrix: np.ndarray):
    """Linear discriminant analysis score function, assuming equal lag priors."""
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)
    
    Cinv_s = np.linalg.solve(covariance_matrix, snippet_c)
    Cinv_t = np.linalg.solve(covariance_matrix, template_c)

    left = np.dot(template_c, Cinv_s)
    right = 0.5 * np.dot(template_c, Cinv_t)

    return float(left - right)

def estimate_r2_lda(snippet: np.ndarray, template: np.ndarray, covariance_matrix: np.ndarray):
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)

    scale = estimate_scale_lda(snippet_c, template_c, covariance_matrix)
    pred = scale * template_c
    resid = snippet_c - pred

    Cinv_resid = np.linalg.solve(covariance_matrix, resid)
    Cinv_s = np.linalg.solve(covariance_matrix, snippet_c)

    sse = float(np.dot(resid, Cinv_resid))
    sst = float(np.dot(snippet_c, Cinv_s))

    if sst <= 1e-20:
        return np.nan
    return float(1.0 - sse / sst)

def estimate_posterior(score_arr: np.ndarray) -> np.ndarray:
    """Answers: Given the feature exists somewhere in this search window, which lag is most likely?"""
    score_arr = np.asarray(score_arr, dtype=float)
    score_arr = score_arr - np.nanmax(score_arr)  # numerical stability, posterior unchanged
    exp_scores = np.exp(score_arr)
    return exp_scores / np.nansum(exp_scores)

def estimate_scale_lda(
    snippet: np.ndarray,
    template: np.ndarray,
    covariance_matrix: np.ndarray,
) -> float:
    if snippet.size != template.size:
        raise ValueError("Snippet and template must have the same length.")
    snippet_c = center_signal(snippet)
    template_c = center_signal(template)
    if covariance_matrix.shape != (snippet_c.size, snippet_c.size):
        raise ValueError("Covariance matrix has wrong shape.")

    # Solve C^{-1} y and C^{-1} t without explicitly inverting
    Cinv_s = np.linalg.solve(covariance_matrix, snippet_c)
    Cinv_t = np.linalg.solve(covariance_matrix, template_c)

    denom = float(np.dot(template_c, Cinv_t))
    if denom <= 1e-20:
        return np.nan

    return float(np.dot(template_c, Cinv_s) / denom)

def build_template_lda(
    intermediate: DataFrame[IntermediateResult],
    template_window: tuple[float, float],
    noise_window: tuple[float, float],
    template_intensities: list[int],
    slope_transform: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Builds a template, covariance matrix, and returns it alongside its slope_transform state."""
    template_data = intermediate.filter(pl.col("intensity").is_in(template_intensities))
    if template_data.is_empty():
        raise ValueError("No traces found for template_intensities.")

    template_snippets, noise_snippets = [], []
    for _, group in template_data.group_by(["id", "intensity"]):
        time = group["time"].to_numpy()
        signal = group["voltage"].to_numpy()

        if slope_transform:
            signal = np.gradient(signal, time)

        template_start, template_stop = window_to_indices(time, template_window)
        noise_start, noise_stop = window_to_indices(time, noise_window)
        template_snippets.append(signal[template_start:template_stop])
        noise_snippets.append(signal[noise_start:noise_stop])

    if template_snippets[-1].size != noise_snippets[-1].size:
        raise ValueError("Noise window and template window must have same number of samples.")

    template_array = np.mean(np.vstack(template_snippets), axis=0)
    covariance_matrix = estimate_noise_covariance(noise_snippets)
    
    return template_array, covariance_matrix, slope_transform


def fit_template_lda(
    intermediate: DataFrame[IntermediateResult],
    template_window: tuple[float, float],
    noise_window: tuple[float, float],
    search_window: tuple[float, float],
    template_package: tuple[np.ndarray, np.ndarray, bool],
    r2_threshold: float,
) -> FeatureResult:
    """Fits a pre-built template package tuple: (template_array, covariance_matrix, slope_transform)"""
    template_arr, covariance_matrix, slope_transform = template_package

    if template_arr.size < 3:
        raise ValueError("Template must contain at least 3 samples.")

    center_idx = int(template_arr.size // 2)
    left = center_idx
    right = template_arr.size - center_idx - 1

    results = []
    for (id_value, intensity), group in intermediate.group_by(["id", "intensity"]):
        time = group["time"].to_numpy()
        signal = group["voltage"].to_numpy()

        if slope_transform:
            signal = np.gradient(signal, time)

        start_idx, stop_idx = window_to_indices(time, search_window)
        first_center = start_idx + left
        last_center = stop_idx - right

        if last_center <= first_center:
            raise ValueError("Search window is too small for this template.")
        
        best_score = -np.inf
        best_result = {
            "id": id_value,
            "intensity": intensity,
            "feature_time": np.nan,
            "scale": np.nan,
            "score": np.nan,
            "score_arr": np.array([]),
            "r2": np.nan,
            "posterior": np.nan,
            "posterior_arr": np.array([]),
            "detected": False,
        }

        score_list = []

        for center in range(first_center, last_center):
            snippet = signal[center - left : center + right + 1]

            scale = estimate_scale_lda(snippet, template_arr, covariance_matrix)
            score = estimate_score(snippet, scale * template_arr, covariance_matrix)

            score_list.append(score)

            if np.isnan(score) or score <= best_score:
                continue

            r2 = estimate_r2_lda(snippet, template_arr, covariance_matrix)
            feature_time = float(time[center] * 1000)
            detected = np.isfinite(r2) and r2 >= r2_threshold

            best_score = score
            best_result = {
                "id": id_value,
                "intensity": intensity,
                "feature_time": feature_time,
                "scale": float(scale),
                "score": float(score),
                "score_arr": np.array([]),
                "r2": float(r2),
                "posterior": np.nan,
                "detected": detected,
            }

        score_arr = np.asarray(score_list, dtype=float)
        best_result["score_arr"] = score_arr

        posterior_arr = estimate_posterior(score_arr)
        best_i = int(np.nanargmax(score_arr))
        best_result["posterior"] = float(posterior_arr[best_i])
        best_result["posterior_arr"] = posterior_arr

        results.append(best_result)
    
    return FeatureResult(
        search_window=search_window,
        template_window=template_window,
        slope_transform=slope_transform,
        r2_threshold=r2_threshold,
        noise_window=noise_window,
        template=template_arr,
        result=pl.DataFrame(results)
    )


def match_feature_lda(
    train_df: DataFrame[IntermediateResult],
    test_df: DataFrame[IntermediateResult],
    template_window: tuple[float, float],
    noise_window: tuple[float, float],
    search_window: tuple[float, float],
    template_intensities: list[int],
    r2_threshold: float,
    slope_transform: bool = False,
) -> FeatureResult:
    """Builds a template from training data and fits it directly onto testing data."""
    template_package = build_template_lda(
        intermediate=train_df,
        template_window=template_window,
        noise_window=noise_window,
        template_intensities=template_intensities,
        slope_transform=slope_transform
    )
    return fit_template_lda(
        intermediate=test_df,
        template_window=template_window,
        noise_window=noise_window,
        search_window=search_window,
        template_package=template_package,
        r2_threshold=r2_threshold,
    )