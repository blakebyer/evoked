# epspkit

## Pipeline
Typical order:
1. Load ABF files into a tidy table.
2. Apply transforms in order.
3. Average sweeps by `stim_intensity`.
4. Run features in the configured order.
5. Save plots and results if enabled.

## IOConfig
Use one of these input patterns:
- `input_paths`: analyze those files directly.
- `template_files` + `test_files`: build templates from `template_files`, then analyze only `test_files`.

Common fields:
- `output_path`
- `stim_intensities`
- `repnum`
- `write_results`
- `write_plots`
- `write_plot_pdf`
- `render_plots`

## Template matching
Template matching is feature-specific and is enabled with `params["method"] = "template"`.

Current behavior:
- Templates are built automatically from `io.template_files`.
- The same preprocessing used on test files is applied to template files first.
- Template building is shared across intensities by default.
- You can restrict template building for a feature with `template_stim_intensities`.
- `window_ms` is the local feature snippet used to build the template.
- `search_window_ms` is the broader region where the feature center is allowed to move during detection.
- The template must fit fully inside `search_window_ms`.

Matching outputs:
- `template_score`: cosine similarity between the candidate snippet and the template.
- `template_scale`: least-squares scale factor for the matched template.

Signal space:
- `fiber_volley`: template match on voltage.
- `epsp`: template match on the first derivative.
- `pop_spike`: template match on voltage.

## Features

### FiberVolleyFeature
Key params:
- `method`: `"peak"` or `"template"`
- `window_ms`: local FV window
- `search_window_ms`: detection region
- `height`: peak-mode trough magnitude threshold in mV
- `template_score_threshold`: optional acceptance threshold in template mode

Outputs:
- `fv_amp`
- `fv_s`
- `fv_v`

### EPSPFeature
Key params:
- `method`: `"peak"` or `"template"`
- `window_ms`: local EPSP snippet window
- `search_window_ms`: detection region
- `fit_distance`: half-width of the linear slope fit
- `height`: peak-mode negative slope threshold in mV/ms
- `template_score_threshold`: optional acceptance threshold in template mode

Outputs:
- `epsp_s`, `epsp_v`
- `slope_mid_s`, `slope_mid_v`
- `epsp_amp`
- `epsp_slope`
- `epsp_r2`
- `epsp_to_fv`

### PopSpikeFeature
Key params:
- `method`: `"peak"` or `"template"`
- `search_window_ms`: detection region for both modes
- `window_ms`: local template/snippet window in template mode
- `height`: peak-mode population spike trough magnitude in mV. For a negative-going spike, use a positive value such as `1.5`.
- `template_score_threshold`: optional acceptance threshold in template mode

Outputs:
- `ps_amp`
- `ps_s`
- `ps_v`

## Transforms
Current transform names:
- `baseline_correction`
- `crop_stim_artifact`
- `template_subtract_stim_artifact`
- `average_sweeps`

Recommended order:
1. `baseline_correction`
2. `crop_stim_artifact` or `template_subtract_stim_artifact`
3. `average_sweeps`

## Plots
Current plot names:
- `sweep`
- `derivative`
- `annotated`
- `template`
- `input_output`

The `template` plot shows:
- the full signal with matched regions highlighted
- the local template fit in actual units
- the correlation profile across the allowed search window

## Example
```python
PipelineConfig(
    io=IOConfig(
        template_files=["/path/to/template.abf"],
        test_files=["/path/to/test.abf"],
        output_path="out/",
        repnum=3,
        stim_intensities=[25, 50, 75, 100, 150, 200],
        write_results=True,
        write_plots=True,
        render_plots=False,
    ),
    transforms=[
        TransformConfig(name="baseline_correction", params={"baseline_window_ms": (0.0, 0.1)}),
        TransformConfig(name="template_subtract_stim_artifact", params={"window_ms": (0.0, 2.25)}),
        TransformConfig(name="average_sweeps"),
    ],
    features=[
        FeatureConfig(
            name="fiber_volley",
            params={
                "method": "template",
                "window_ms": (1.5, 3.0),
                "search_window_ms": (1.0, 3.2),
                "template_score_threshold": 0.8,
            },
        ),
        FeatureConfig(
            name="epsp",
            params={
                "method": "template",
                "window_ms": (2.5, 4.0),
                "search_window_ms": (1.5, 6.0),
                "fit_distance": 4,
                "template_score_threshold": 0.8,
            },
        ),
        FeatureConfig(
            name="pop_spike",
            params={
                "method": "peak",
                "search_window_ms": (4.0, 8.0),
                "height": 1.5,
            },
        ),
    ],
    plots=[VizConfig(name="template")],
)
```
