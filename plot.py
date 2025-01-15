import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(results_df: pd.DataFrame) -> plt.Figure:
    """Plot benchmark results with separate subplot per dtype."""
    backends = results_df.columns.levels[0]
    dtypes = results_df.columns.levels[1]

    fig, axes = plt.subplots(
        len(dtypes),
        1,
        figsize=(12, 4 * len(dtypes)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    colors = plt.cm.Set3(np.linspace(0, 1, len(backends)))
    bar_width = 0.8 / len(backends)
    x = np.arange(len(results_df.index))

    for ax, dtype in zip(axes, dtypes):
        ax: plt.Axes
        dtype: str

        data = pd.DataFrame({b: results_df[b][dtype] for b in backends})
        best_perf = data.max(axis=1)

        for i, (backend, color) in enumerate(zip(backends, colors)):
            values = data[backend]
            if values.isnull().all():
                continue

            bars = ax.bar(
                x + i * bar_width,
                values,
                bar_width,
                label=backend,
                color=color,
            )

            best_mask = np.isclose(values, best_perf, rtol=1e-5)
            for bar, is_best in zip(bars, best_mask):
                bar: plt.Rectangle
                if is_best and not np.isnan(bar.get_height()):
                    ax.bar(
                        bar.get_x(),
                        bar.get_height(),
                        bar_width,
                        label="_nolegend_",
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                        align="edge",
                    )

        ax.set_title(f"{dtype=}")
        ax.set_ylabel("TFLOPS")
        ax.grid(True, axis="y", linestyle="--")
        ax.set_xticks(x + bar_width * len(backends) / 2)
        ax.set_xticklabels(results_df.index, rotation=30, ha="right")

    axes[0].legend(title="backend", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[-1].set_xlabel("shape")
    fig.suptitle("matmul-tlops-comparison", y=1.02)
    fig.tight_layout()

    return fig
