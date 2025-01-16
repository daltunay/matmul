import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from implementations import BACKENDS


def process_results(results_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = results_df.copy()
    plot_df["shape"] = (
        "("
        + plot_df["M"].astype(str)
        + ", "
        + plot_df["N"].astype(str)
        + ", "
        + plot_df["K"].astype(str)
        + ")"
    )

    plot_df = plot_df.pivot(
        index="shape",
        columns=["dtype"],
        values=list(BACKENDS.keys()),
    )

    return plot_df


def plot_results(results_df: pd.DataFrame) -> plt.Figure:
    """Plot benchmark results with separate subplot per dtype."""
    plot_df = process_results(results_df)

    backends = plot_df.columns.levels[0]
    dtypes = plot_df.columns.levels[1]

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
    x = np.arange(len(plot_df.index))

    for ax, dtype in zip(axes, dtypes):
        ax: plt.Axes
        dtype: str

        data = pd.DataFrame({b: plot_df[b][dtype] for b in backends})

        for i, (backend, color) in enumerate(zip(backends, colors)):
            values = data[backend]
            if values.isnull().all():
                continue

            ax.bar(
                x + i * bar_width,
                values,
                bar_width,
                label=backend,
                color=color,
            )

        ax.set_title(f"{dtype=}")
        ax.set_ylabel("TFLOPS")
        ax.grid(True, axis="y", linestyle="--")
        ax.set_xticks(x + bar_width * len(backends) / 2)
        ax.set_xticklabels(plot_df.index, rotation=30, ha="right")

    axes[0].legend(title="backend", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[-1].set_xlabel("shape")
    fig.suptitle("matmul-tlops-comparison", y=1.02)
    fig.tight_layout()

    return fig
