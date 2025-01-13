import pandas as pd
import plotly.graph_objects as go


def create_grid_lines(df: pd.DataFrame) -> list[go.Scatter3d]:
    """Create 3D grid lines connecting all unique M, N, K combinations."""
    unique_m = sorted(df["M"].unique())
    unique_n = sorted(df["N"].unique())
    unique_k = sorted(df["K"].unique())

    lines = []

    for n in unique_n:
        for k in unique_k:
            if len(df[(df["N"] == n) & (df["K"] == k)]) > 0:
                lines.append(
                    go.Scatter3d(
                        x=unique_m,
                        y=[n] * len(unique_m),
                        z=[k] * len(unique_m),
                        mode="lines",
                        line=dict(color="black", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    for m in unique_m:
        for k in unique_k:
            if len(df[(df["M"] == m) & (df["K"] == k)]) > 0:
                lines.append(
                    go.Scatter3d(
                        x=[m] * len(unique_n),
                        y=unique_n,
                        z=[k] * len(unique_n),
                        mode="lines",
                        line=dict(color="black", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    for m in unique_m:
        for n in unique_n:
            if len(df[(df["M"] == m) & (df["N"] == n)]) > 0:
                lines.append(
                    go.Scatter3d(
                        x=[m] * len(unique_k),
                        y=[n] * len(unique_k),
                        z=unique_k,
                        mode="lines",
                        line=dict(color="black", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    return lines


def create_3d_plot(df: pd.DataFrame) -> go.Figure:
    """Create a single 3D scatter plot with different markers for each backend."""
    backend_cols = [col for col in df.columns if col not in ["M", "N", "K", "dtype"]]

    fig = go.Figure()

    for line in create_grid_lines(df):
        fig.add_trace(line)

    markers = ["circle", "square", "diamond", "cross", "x"]

    for idx, backend in enumerate(backend_cols):
        scatter = go.Scatter3d(
            x=df["M"],
            y=df["N"],
            z=df["K"],
            mode="markers",
            name=backend,
            marker=dict(
                size=10,
                symbol=markers[idx % len(markers)],
                color=df[backend],
                colorscale="Viridis",
                colorbar=dict(
                    title="TFLOPS",
                    x=1.0,
                    xanchor="left",
                ),
                showscale=(idx == len(backend_cols) - 1),
            ),
            text=[
                f"{backend}<br>M={m}, N={n}, K={k}<br>TFLOPS={v:.2f}"
                for m, n, k, v in zip(df["M"], df["N"], df["K"], df[backend])
            ],
            hoverinfo="text",
        )
        fig.add_trace(scatter)

    fig.update_layout(
        title="Matrix Multiplication Performance Comparison",
        scene=dict(
            xaxis_title="M",
            yaxis_title="N",
            zaxis_title="K",
            xaxis_type="log",
            yaxis_type="log",
            zaxis_type="log",
        ),
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    return fig


if __name__ == "__main__":
    results_df = pd.read_csv("results/matmul-tflops-comparison.csv")
    fig = create_3d_plot(results_df)
    fig.show()
