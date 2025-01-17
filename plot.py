import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_hover_template(performance_line: str) -> str:
    """Create a hover template with the given performance line."""
    return (
        "<b>%{data.name}</b><br>"
        "<br>"
        "<b>Hardware:</b><br>"
        "Device: %{customdata[8]}<br>"
        "Device Count: %{customdata[0]}<br>"
        "<br>"
        "<b>Shape:</b><br>"
        "M=%{customdata[2]}, N=%{customdata[3]}, K=%{customdata[4]}<br>"
        "<br>"
        "<b>Performance:</b><br>"
        f"{performance_line}"
        "<br>"
        "<b>Software:</b><br>"
        "Python: %{customdata[5]}<br>"
        "PyTorch: %{customdata[6]}<br>"
        "CUDA: %{customdata[7]}<br>"
        "<extra></extra>"
    )


time_hover = create_hover_template(
    "Time: %{y:.3f} ms<br>TFLOP/s: %{customdata[1]:.3f}<br>"
)
time_norm_hover = create_hover_template(
    "Time per Device: %{y:.3f} ms<br>TFLOP/s: %{customdata[1]:.3f}<br>"
)
tflops_hover = create_hover_template(
    "Time: %{customdata[1]:.3f} ms<br>TFLOP/s: %{y:.3f}<br>"
)
tflops_norm_hover = create_hover_template(
    "Time: %{customdata[1]:.3f} ms<br>TFLOP/s per Device: %{y:.3f}<br>"
)


def create_button(state_args: dict, label: str) -> dict:
    """Create a button with trace and layout updates."""
    return dict(
        args=[
            {
                key: state_args[key]
                for key in ["y", "visible", "hovertemplate"]
                if key in state_args
            },
            {
                key: state_args[key]
                for key in ["barmode", "yaxis.title", "yaxis.range"]
                if key in state_args
            },
        ],
        label=label,
        method="update",
    )


def create_menu(buttons: list, x: float) -> dict:
    """Create a menu configuration dictionary."""
    return dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=x,
        xanchor="left",
        y=1.15,
        yanchor="bottom",
        pad=dict(t=0, b=0),
        bgcolor="rgba(255, 255, 255, 0.9)",
    )


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


def create_figure(df: pd.DataFrame, normalize: bool = False) -> go.Figure:
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

    time_data = []
    flops_data = []
    shapes = []
    device_dtype_groups = {}

    backend_traces = {backend: [] for backend in backends}

    for device_idx, device in enumerate(devices, 1):
        for dtype_idx, dtype in enumerate(dtypes, 1):
            group_key = (device_idx, dtype_idx)
            device_dtype_groups[group_key] = []

            for backend in backends:
                mask = (
                    (df["dtype"] == dtype)
                    & (df["device_name"] == device)
                    & (df["backend"] == backend)
                )
                if not any(mask):
                    continue

                device_data = df[mask].copy()
                trace_idx = len(time_data)
                device_dtype_groups[group_key].append(trace_idx)
                backend_traces[backend].append(trace_idx)

                time_data.append(device_data["time_ms"].tolist())
                flops_data.append(device_data["tflop/s"].tolist())
                shapes.append(device_data["shape"].tolist())

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
                                "device_name",
                            ]
                        ],
                        visible=True,
                        showlegend=(device_idx == 1 and dtype_idx == 1),
                        marker_color=backend_colors[backend],
                        legendgroup=backend,
                        hovertemplate=(
                            "<b>%{data.name}</b><br>"
                            "<br>"
                            "<b>Hardware:</b><br>"
                            "Device: %{customdata[8]}<br>"
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

    if normalize:
        time_data = [
            [t / df.iloc[i]["device_count"] for t in times]
            for i, times in enumerate(time_data)
        ]
        flops_data = [
            [f / df.iloc[i]["device_count"] for f in flops]
            for i, flops in enumerate(flops_data)
        ]

    def get_hover_template(is_time: bool, is_normalized: bool) -> str:
        if is_time:
            return time_norm_hover if is_normalized else time_hover
        return tflops_norm_hover if is_normalized else tflops_hover

    def calculate_percentages(data):
        """Convert data to percentages within each device-dtype group"""
        percentage_data = []
        for group_indices in device_dtype_groups.values():
            if not group_indices:
                continue
            for shape_idx in range(len(shapes[group_indices[0]])):
                shape_values = [data[idx][shape_idx] for idx in group_indices]
                total = sum(shape_values)
                if total > 0:
                    percentages = [100 * val / total for val in shape_values]
                else:
                    percentages = [0] * len(shape_values)
                for idx, perc in zip(group_indices, percentages):
                    if len(percentage_data) <= idx:
                        percentage_data.append([])
                    percentage_data[idx].append(perc)
        return percentage_data

    time_pct = calculate_percentages(time_data)
    flops_pct = calculate_percentages(flops_data)

    current_state = {
        "is_time": True,
        "is_normalized": False,
        "view_mode": "group",
    }

    def get_y_data(is_time: bool, view_mode: str):
        """Get appropriate y data based on current mode"""
        if view_mode == "stack_percent":
            return time_pct if is_time else flops_pct
        return time_data if is_time else flops_data

    def get_state_updates(is_time: bool, view_mode: str) -> dict:
        """Get complete state updates for buttons."""
        data = {
            "time": time_data if view_mode != "stack_percent" else time_pct,
            "flops": flops_data if view_mode != "stack_percent" else flops_pct,
        }
        y_data = data["time"] if is_time else data["flops"]

        template = get_hover_template(is_time, normalize)
        if view_mode == "stack_percent":
            template = template.replace("Time:", "Percentage:").replace(" ms", "%")

        metric = (
            "Percentage (%)"
            if view_mode == "stack_percent"
            else (
                f"Time{' per Device' if normalize else ''} (ms)"
                if is_time
                else f"TFLOP/s{' per Device' if normalize else ''}"
            )
        )

        updates = {
            "y": y_data,
            "visible": [True] * len(fig.data),
            "hovertemplate": [template] * len(fig.data),
            "barmode": "stack" if "stack" in view_mode else "group",
            "yaxis.title": metric,
            "xaxis.title": "Matrix Shape",
        }

        if view_mode == "stack_percent":
            updates["yaxis.range"] = [0, 100]

        return updates

    view_buttons = [
        create_button(
            get_state_updates(current_state["is_time"], mode),
            label,
        )
        for mode, label in [
            ("group", "Grouped"),
            ("stack", "Stacked (Absolute)"),
            ("stack_percent", "Stacked (%)"),
        ]
    ]

    metric_buttons = [
        create_button(
            get_state_updates(is_time, current_state["view_mode"]),
            "Time (ms)" if is_time else "TFLOP/s",
        )
        for is_time in [True, False]
    ]

    updatemenus = [
        create_menu(metric_buttons, x=0.2),
        create_menu(view_buttons, x=0.6),
    ]

    min_height = 300
    max_height = 500
    total_height = min(max(min_height * len(devices), 600), max_height * len(devices))

    fig.update_layout(
        height=total_height,
        width=max(1200, 400 * len(dtypes)),
        title_text=f"Matrix Multiplication Performance{' (Normalized)' if normalize else ''}",
        title_y=0.98,
        showlegend=True,
        barmode="group",
        bargap=0.15,
        bargroupgap=0.1,
        updatemenus=updatemenus,
        margin=dict(t=120),
        legend=dict(
            groupclick="toggleitem",
        ),
        yaxis_title=f"Time{' per Device' if normalize else ''} (ms)",
        xaxis_title="Matrix Shape",
    )

    for i in range(1, len(devices) * len(dtypes) + 1):
        fig.update_xaxes(
            title_text="Matrix Shape",
            row=i // len(dtypes) + 1,
            col=i % len(dtypes) or len(dtypes),
        )
        fig.update_yaxes(
            title_text=f"Time{' per Device' if normalize else ''} (ms)",
            row=i // len(dtypes) + 1,
            col=i % len(dtypes) or len(dtypes),
        )

    return fig


def plot_benchmarks(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    """Create both regular and normalized benchmark plots."""
    return create_figure(df, normalize=False), create_figure(df, normalize=True)
