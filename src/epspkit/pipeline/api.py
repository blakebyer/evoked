from __future__ import annotations

from collections.abc import Sequence
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages

from epspkit.core.config import (
    FeatureConfig,
    PipelineConfig,
    SmoothingConfig,
    TransformConfig,
    VizConfig,
)
from epspkit.core.context import RecordingContext
from epspkit.features.base import apply_smoothing
from epspkit.features.epsp import EPSPFeature
from epspkit.features.fiber_volley import FiberVolleyFeature
from epspkit.features.pop_spike import PopSpikeFeature
from epspkit.io.read_write import load_abf_to_context, save_context_to_xlsx
from epspkit.transforms.average import average_sweeps
from epspkit.transforms.baseline import baseline_correction
from epspkit.transforms.stim_artifact import (
    crop_stim_artifact,
    template_subtract_stim_artifact,
)
from epspkit.transforms.template import average_templates, build_template
from epspkit.transforms.template import capture_template_window
from epspkit.viz.annotated import AnnotatedPlot
from epspkit.viz.derivative import DerivativePlot
from epspkit.viz.input_output import IOPlot
from epspkit.viz.sweep import SweepPlot
from epspkit.viz.template import TemplatePlot
from epspkit.viz.template_simple import TemplateSimplePlot

FEATURE_CLASSES: dict[str, type] = {
    "fiber_volley": FiberVolleyFeature,
    "epsp": EPSPFeature,
    "pop_spike": PopSpikeFeature,
}

PLOT_CLASSES: dict[str, type] = {
    "sweep": SweepPlot,
    "derivative": DerivativePlot,
    "annotated": AnnotatedPlot,
    "input_output": IOPlot,
    "template": TemplatePlot,
    "template_simple": TemplateSimplePlot,
}

TRANSFORM_FUNCS: dict[str, Callable[..., RecordingContext]] = {
    "baseline_correction": baseline_correction,
    "crop_stim_artifact": crop_stim_artifact,
    "template_subtract_stim_artifact": template_subtract_stim_artifact,
    "average_sweeps": average_sweeps,
    "capture_template_window": capture_template_window,
}

def effective_smoothing(
    config: FeatureConfig | VizConfig,
    global_smoothing: SmoothingConfig,
) -> SmoothingConfig:
    return config.smoothing if config.smoothing is not None else global_smoothing


def _build_component(
    config: FeatureConfig | VizConfig,
    registry: dict[str, type],
    kind: str,
    global_smoothing: SmoothingConfig,
):
    component_cls = registry.get(config.name)
    if component_cls is None:
        options = ", ".join(sorted(registry))
        raise ValueError(f"Unknown {kind} '{config.name}'. Available: {options}")
    smoothing = effective_smoothing(config, global_smoothing)
    return component_cls(config, smoothing)


def build_feature(config: FeatureConfig, global_smoothing: SmoothingConfig):
    return _build_component(config, FEATURE_CLASSES, "feature", global_smoothing)


def build_plot(config: VizConfig, global_smoothing: SmoothingConfig):
    return _build_component(config, PLOT_CLASSES, "plot", global_smoothing)


def _resolve_plot_pdf_path(pipeline_config: PipelineConfig) -> Path:
    io_cfg = pipeline_config.io
    if io_cfg.plot_pdf_path is not None:
        output = Path(io_cfg.plot_pdf_path)
        if not output.suffix:
            output = output / "run_plots.pdf"
    elif io_cfg.output_path is not None:
        output = Path(io_cfg.output_path)
        if output.suffix:
            output = output.with_name(f"{output.stem}_plots.pdf")
        else:
            output = output / "run_plots.pdf"
    else:
        raise ValueError(
            "A plot PDF requires PipelineConfig.io.output_path or plot_pdf_path."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _plot_pdf_grid(n_plots: int) -> tuple[int, int]:
    if n_plots < 1:
        raise ValueError("n_plots must be at least 1.")
    cols = min(3, max(1, ceil(sqrt(n_plots))))
    rows = ceil(n_plots / cols)
    return rows, cols


def _build_plot_figure(plot: Any, context: RecordingContext):
    builder = getattr(plot, "build_figure", None)
    if builder is None:
        builder = getattr(plot, "_build_figure", None)
    if builder is None:
        raise AttributeError(
            f"Plot '{plot.name}' does not expose a figure-building method."
        )
    fig = builder(context)
    if fig is None:
        raise ValueError(f"Plot '{plot.name}' did not return a Matplotlib figure.")
    return fig


def _figure_to_rgba(fig: Any) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba()).copy()


def _save_plot_pdf_page(
    pdf: PdfPages,
    page_items: Sequence[tuple[np.ndarray, str]],
) -> None:
    rows, cols = _plot_pdf_grid(len(page_items))
    page_fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(11, 8.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (image, title) in zip(axes, page_items):
        ax.imshow(image)
        ax.set_title(title, fontsize=10, pad=8)
        ax.axis("off")

    for ax in axes[len(page_items):]:
        ax.axis("off")

    pdf.savefig(page_fig)
    plt.close(page_fig)


def write_plot_pdf(
    contexts: Sequence[RecordingContext],
    pipeline_config: PipelineConfig,
    output_stems: Sequence[str] | None = None,
) -> Path | None:
    plot_cfgs = list(pipeline_config.plots or [])
    if not plot_cfgs:
        return None

    plots_per_page = int(pipeline_config.io.plot_pdf_plots_per_page)
    if plots_per_page < 1:
        raise ValueError("plot_pdf_plots_per_page must be at least 1.")

    pdf_path = _resolve_plot_pdf_path(pipeline_config)
    stems = list(output_stems or [])
    if not stems:
        stems = [f"recording_{idx + 1}" for idx in range(len(contexts))]

    page_items: list[tuple[np.ndarray, str]] = []
    with PdfPages(pdf_path) as pdf:
        for idx, context in enumerate(contexts):
            stem = stems[idx] if idx < len(stems) else f"recording_{idx + 1}"
            for plot_cfg in plot_cfgs:
                plot = build_plot(plot_cfg, pipeline_config.global_smoothing)
                fig = _build_plot_figure(plot, context)
                try:
                    image = _figure_to_rgba(fig)
                finally:
                    plt.close(fig)

                page_items.append((image, f"{stem} | {plot_cfg.name}"))
                if len(page_items) == plots_per_page:
                    _save_plot_pdf_page(pdf, page_items)
                    page_items = []

        if page_items:
            _save_plot_pdf_page(pdf, page_items)

    return pdf_path


def resolve_plot_smoothing(pipeline_config: PipelineConfig) -> None:
    if not pipeline_config.plots:
        return
    global_cfg = pipeline_config.global_smoothing
    for plot_cfg in pipeline_config.plots:
        if plot_cfg.smoothing is None:
            plot_cfg.smoothing = SmoothingConfig(
                method=global_cfg.method,
                window_size=global_cfg.window_size,
                polyorder=global_cfg.polyorder,
                cutoff=global_cfg.cutoff,
                order=global_cfg.order,
            )


def resolve_feature_smoothing(pipeline_config: PipelineConfig) -> None:
    if not pipeline_config.features:
        return
    global_cfg = pipeline_config.global_smoothing
    for feature_cfg in pipeline_config.features:
        if feature_cfg.smoothing is None:
            feature_cfg.smoothing = SmoothingConfig(
                method=global_cfg.method,
                window_size=global_cfg.window_size,
                polyorder=global_cfg.polyorder,
                cutoff=global_cfg.cutoff,
                order=global_cfg.order,
            )

def build_transform(
    config: TransformConfig,
) -> tuple[Callable[..., RecordingContext], dict[str, Any]]:
    transform_fn = TRANSFORM_FUNCS.get(config.name)
    if transform_fn is None:
        options = ", ".join(sorted(TRANSFORM_FUNCS))
        raise ValueError(f"Unknown transform '{config.name}'. Available: {options}")
    return transform_fn, config.params or {}


def prepare_context(
    context: RecordingContext,
    pipeline_config: PipelineConfig,
) -> RecordingContext:
    context.pipeline_cfg = pipeline_config
    resolve_plot_smoothing(pipeline_config)
    resolve_feature_smoothing(pipeline_config)

    for transform_cfg in pipeline_config.transforms:
        transform_fn, params = build_transform(transform_cfg)
        result = transform_fn(context, **params)
        if result is not None and not isinstance(result, RecordingContext):
            raise TypeError("Transforms must return RecordingContext or None.")
        if isinstance(result, RecordingContext):
            context = result

    if context.averaged is None or context.averaged.empty:
        average_sweeps(context)

    return context


def run_context(
    context: RecordingContext,
    pipeline_config: PipelineConfig,
    output_stem: str | None = None,
) -> RecordingContext:
    if pipeline_config.io.write_plots and not pipeline_config.io.output_path:
        raise ValueError(
            "PipelineConfig.io.output_path is required when write_plots is True."
        )

    context = prepare_context(context, pipeline_config)

    for feature_cfg in pipeline_config.features:
        feature = build_feature(feature_cfg, pipeline_config.global_smoothing)
        context = feature.run(context)

    plots = pipeline_config.plots
    if plots and (pipeline_config.io.render_plots or pipeline_config.io.write_plots):
        for plot_cfg in plots:
            plot = build_plot(plot_cfg, pipeline_config.global_smoothing)
            if pipeline_config.io.render_plots:
                plot.render(context)
            if pipeline_config.io.write_plots:
                plot.save(context, pipeline_config.io.output_path, output_stem=output_stem)

    return context


def _uses_template_method(feature_cfg: FeatureConfig) -> bool:
    method = str(feature_cfg.params.get("method", "derivative")).lower()
    return method == "template"


def _resolve_test_paths(pipeline_config: PipelineConfig) -> list[str]:
    io_cfg = pipeline_config.io
    return list(io_cfg.test_files or io_cfg.input_files)


def _load_context(path: str, pipeline_config: PipelineConfig) -> RecordingContext:
    context = load_abf_to_context(
        file_path=path,
        stim_intensities=list(pipeline_config.io.stim_intensities),
        repnum=pipeline_config.io.repnum,
    )
    context.metadata.update(pipeline_config.io.metadata or {})
    return context


def _build_template_for_feature(
    feature_cfg: FeatureConfig,
    template_contexts: Sequence[RecordingContext],
    global_smoothing: SmoothingConfig,
) -> tuple[np.ndarray, int]:
    if feature_cfg.name not in {"fiber_volley", "epsp", "pop_spike"}:
        raise ValueError(
            f"Automatic template building is not implemented for feature "
            f"'{feature_cfg.name}'."
        )

    window_ms = feature_cfg.params.get("window_ms")
    if window_ms is None:
        raise ValueError(
            f"Feature '{feature_cfg.name}' requires 'window_ms' to build a template."
        )
    template_stim_intensities = feature_cfg.params.get("template_stim_intensities")

    center_idx = feature_cfg.params.get("template_center_idx")
    snippets: list[np.ndarray] = []

    for context in template_contexts:
        frame = context.averaged
        if template_stim_intensities is not None:
            if "stim_intensity" not in frame.columns:
                raise ValueError(
                    f"Feature '{feature_cfg.name}' requested template_stim_intensities, "
                    "but averaged data has no 'stim_intensity' column."
                )
            frame = frame.loc[frame["stim_intensity"].isin(template_stim_intensities)]
            if frame.empty:
                continue

        for _, g in frame.groupby("stim_intensity"):
            x = g["time"].to_numpy()
            y = apply_smoothing(
                g["mean"].to_numpy(),
                feature_cfg.smoothing or global_smoothing,
                fs=context.fs,
            )

            snippet, resolved_center = build_template(
                x,
                y,
                window_ms,
                center_idx=center_idx,
                feature_name=feature_cfg.name,
            )
            if center_idx is None:
                center_idx = resolved_center
            snippets.append(snippet)

    if not snippets:
        if template_stim_intensities is not None:
            raise ValueError(
                f"No template snippets were produced for feature '{feature_cfg.name}' "
                f"using template_stim_intensities={list(template_stim_intensities)}."
            )
        raise ValueError(
            f"No template snippets were produced for feature '{feature_cfg.name}'."
        )

    return average_templates(snippets), int(center_idx)


def _inject_built_templates(pipeline_config: PipelineConfig) -> None:
    template_feature_cfgs = [cfg for cfg in pipeline_config.features if _uses_template_method(cfg)]
    if not template_feature_cfgs:
        return

    template_files = list(pipeline_config.io.template_files)
    if not template_files:
        raise ValueError(
            "PipelineConfig.io.template_files is required when any feature "
            "uses method='template'."
        )

    template_contexts = [
        prepare_context(_load_context(path, pipeline_config), pipeline_config)
        for path in template_files
    ]

    for feature_cfg in template_feature_cfgs:
        template, center_idx = _build_template_for_feature(
            feature_cfg,
            template_contexts,
            pipeline_config.global_smoothing,
        )
        feature_cfg.params["template"] = template.tolist()
        feature_cfg.params["template_center_idx"] = center_idx


def run_pipeline(
    pipeline_config: PipelineConfig,
) -> list[RecordingContext]:
    test_paths = _resolve_test_paths(pipeline_config)
    if not test_paths:
        raise ValueError(
            "PipelineConfig.io.test_files or input_files is required."
        )
    if not pipeline_config.io.stim_intensities:
        raise ValueError("PipelineConfig.io.stim_intensities is required.")
    if (
        pipeline_config.io.write_results
        or pipeline_config.io.write_plots
    ) and not pipeline_config.io.output_path:
        raise ValueError(
            "PipelineConfig.io.output_path is required when write_results or "
            "write_plots is True."
        )
    if (
        pipeline_config.io.write_plot_pdf
        and pipeline_config.io.output_path is None
        and pipeline_config.io.plot_pdf_path is None
    ):
        raise ValueError(
            "PipelineConfig.io.output_path or plot_pdf_path is required when "
            "write_plot_pdf is True."
        )

    _inject_built_templates(pipeline_config)

    def build_output_stems(paths: Sequence[str]) -> list[str]:
        return [Path(path).stem for path in paths]

    output_stems = build_output_stems(test_paths)
    contexts = []
    for idx, path in enumerate(test_paths):
        context = _load_context(path, pipeline_config)
        output_stem = output_stems[idx] if idx < len(output_stems) else None
        contexts.append(run_context(context, pipeline_config, output_stem=output_stem))

    if pipeline_config.io.write_results and pipeline_config.io.output_path:
        write_results(contexts, pipeline_config.io.output_path, output_stems)
    if pipeline_config.io.write_plot_pdf:
        write_plot_pdf(contexts, pipeline_config, output_stems)

    return contexts


def write_results(
    contexts: Sequence[RecordingContext],
    output_path: Path | str,
    output_stems: Sequence[str] | None = None,
) -> None:
    output = Path(output_path)
    stems = list(output_stems or [])
    if not stems:
        stems = [f"recording_{idx + 1}" for idx in range(len(contexts))]

    if output.suffix and len(contexts) == 1:
        save_context_to_xlsx(contexts[0], str(output), output_stem=stems[0])
        return

    output_dir = output if not output.suffix else output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, context in enumerate(contexts):
        stem = stems[idx] if idx < len(stems) else f"recording_{idx + 1}"
        save_context_to_xlsx(context, str(output_dir), output_stem=stem)
