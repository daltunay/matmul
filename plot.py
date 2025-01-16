import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def process_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Process results DataFrame for plotting."""
    results_df = results_df.copy()
    results_df["shape"] = (
        "("
        + results_df["M"].astype(str)
        + ", "
        + results_df["N"].astype(str)
        + ", "
        + results_df["K"].astype(str)
        + ")"
    )

    plot_df = results_df.pivot_table(
        index="shape", columns=["dtype", "backend"], values="tflop/s", fill_value=None
    )

    return plot_df


def create_figure(df: pd.DataFrame) -> go.Figure:
    """Create interactive plotly figure for benchmark results."""
    dtypes = sorted(df["dtype"].unique())
    devices = sorted(df["device_name"].unique())
    backends = sorted(df["backend"].unique())

    color_scale = plt.cm.Set3(np.linspace(0, 1, len(backends)))
    backend_colors = {
        backend: f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        for backend, (r, g, b, _) in zip(backends, color_scale)
    }

    fig = make_subplots(
        rows=len(devices),
        cols=len(dtypes),
        subplot_titles=[f"{dev} - {dt}" for dev in devices for dt in dtypes],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
    )

    df["shape"] = (
        "("
        + df["M"].astype(str)
        + ","
        + df["N"].astype(str)
        + ","
        + df["K"].astype(str)
        + ")"
    )

    for device_idx, device in enumerate(devices, 1):
        for dtype_idx, dtype in enumerate(dtypes, 1):
            for backend in backends:
                mask = (
                    (df["dtype"] == dtype)
                    & (df["device_name"] == device)
                    & (df["backend"] == backend)
                )
                if not any(mask):
                    continue

                device_data = df[mask].copy()

                fig.add_trace(
                    go.Bar(
                        name=backend,
                        x=device_data["shape"],
                        y=device_data["time_ms"],
                        customdata=device_data[
                            [
                                "device_count",
                                "tflop/s",
                                "M",
                                "N",
                                "K",
                                "python_version",
                                "torch_version",
                                "cuda_version",
                                "device_name"  # Add device_name to customdata
                            ]
                        ],
                        visible=True,
                        showlegend=(device_idx == 1 and dtype_idx == 1),
                        marker_color=backend_colors[backend],
                        hovertemplate=(
                            "<b>%{data.name}</b><br>"
                            "<br>"
                            "<b>Hardware:</b><br>"
                            "Device: %{customdata[8]}<br>"  # Show device name
                            "Device Count: %{customdata[0]}<br>"
                            "<br>"
                            "<b>Shape:</b><br>"
                            "M=%{customdata[2]}, N=%{customdata[3]}, K=%{customdata[4]}<br>"
                            "<br>"
                            "<b>Performance:</b><br>"
                            "Time: %{y:.3f} ms<br>"
                            "TFLOP/s: %{customdata[1]:.3f}<br>"
                            "<br>"
                            "<b>Software:</b><br>"
                            "Python: %{customdata[5]}<br>"
                            "PyTorch: %{customdata[6]}<br>"
                            "CUDA: %{customdata[7]}<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=device_idx,
                    col=dtype_idx,
                )

            fig.update_xaxes(
                tickangle=45,
                tickmode="array",
                ticktext=device_data["shape"],
                tickvals=list(range(len(device_data["shape"]))),
                row=device_idx,
                col=dtype_idx,
            )

    # Template strings for different states
    time_template = (
        "<b>Performance:</b><br>"
        "Time: %{y:.3f} ms<br>"
        "TFLOP/s: %{customdata[1]:.3f}"
    )
    time_norm_template = (
        "<b>Performance:</b><br>"
        "Time per Device: %{y:.3f} ms<br>"
        "TFLOP/s: %{customdata[1]:.3f}"
    )
    tflops_template = (
        "<b>Performance:</b><br>"
        "TFLOP/s: %{customdata[1]:.3f}<br>"
        "Time: %{y:.3f} ms"
    )
    tflops_norm_template = (
        "<b>Performance:</b><br>"
        "TFLOP/s per Device: %{customdata[1]:.3f}<br>"
        "Time: %{y:.3f} ms"
    )

    # Base hover template with sections
    base_hover_template = (
        "<b>%{data.name}</b><br>"
        "<br>"
        "<b>Hardware:</b><br>"
        "Device: %{customdata[8]}<br>"
        "Device Count: %{customdata[0]}<br>"
        "<br>"
        "<b>Shape:</b><br>"
        "M=%{customdata[2]}, N=%{customdata[3]}, K=%{customdata[4]}<br>"
        "<br>"
        "{metric}<br>"
        "<br>"
        "<b>Software:</b><br>"
        "Python: %{customdata[5]}<br>"
        "PyTorch: %{customdata[6]}<br>"
        "CUDA: %{customdata[7]}<br>"
        "<extra></extra>"
    )

    current_metric = {"is_time": True, "is_normalized": False}

    def get_hover_template(is_time: bool, is_normalized: bool) -> str:
        if is_time:
            return time_norm_hover if is_normalized else time_hover
        return tflops_norm_hover if is_normalized else tflops_hover

    metric_buttons = [
        dict(
            args=[
                {
                    "visible": [True] * len(fig.data),
                    "hovertemplate": [time_hover] * len(fig.data),
                },
                {"yaxis.title": "Time (ms)"},
            ],
            label="Time (ms)",
            method="update",
        ),
        dict(
            args=[
                {
                    "visible": [True] * len(fig.data),
                    "hovertemplate": [tflops_hover] * len(fig.data),
                },
                {"yaxis.title": "TFLOP/s"},
            ],
            label="TFLOP/s",
            method="update",
        ),
    ]

    norm_buttons = [
        dict(
            args=[
                {
                    "visible": [True] * len(fig.data),
                    "hovertemplate": [
                        get_hover_template(current_metric["is_time"], False)
                    ]
                    * len(fig.data),
                }
            ],
            label="Raw Values",
            method="update",
        ),
        dict(
            args=[
                {
                    "visible": [True] * len(fig.data),
                    "hovertemplate": [
                        get_hover_template(current_metric["is_time"], True)
                    ]
                    * len(fig.data),
                }
            ],
            label="Normalized by Device Count",
            method="update",
        ),
    ]

    updatemenus = [
        dict(
            buttons=metric_buttons,
            direction="down",
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top",
        ),
        dict(
            buttons=norm_buttons,
            direction="down",
            showactive=True,
            x=0.3,
            xanchor="left",
            y=1.15,
            yanchor="top",
        ),
        dict(
            buttons=list(
                [
                    dict(
                        args=[{"barmode": "group"}],
                        label="Grouped Bars",
                        method="relayout",
                    ),
                    dict(
                        args=[{"barmode": "stack"}],
                        label="Stacked Bars",
                        method="relayout",
                    ),
                ]
            ),
            direction="down",
            showactive=True,
            x=0.5,
            xanchor="left",
            y=1.15,
            yanchor="top",
        ),
    ]

    fig.update_layout(
        height=400 * len(devices),
        width=max(1200, 400 * len(dtypes)),
        title_text="Matrix Multiplication Performance",
        showlegend=True,
        hovermode="x",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.1,
        updatemenus=updatemenus,
    )


    return fig
