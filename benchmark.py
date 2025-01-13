import json

import pandas as pd
import torch
import triton
import triton.testing

from implementations import BACKENDS, MatrixBackend
from plot import create_3d_plot

with open("matrix_specs.json", "r") as f:
    # real-world (M, N, K) shapes for Llama 70B production training
    # https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/#general-matrix-multiply-gemm-performance
    # https://github.com/pytorch-labs/float8_experimental/blob/main/benchmarks/bench_matmul.py#L58
    MATRIX_SHAPES = json.load(f)


def get_device():
    """Get the appropriate device for benchmarking."""
    if torch.cuda.is_available():
        torch.cuda.init()
        return "cuda"
    print("WARNING: CUDA not available, falling back to CPU")
    return "cpu"


configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[(c["M"], c["N"], c["K"]) for c in MATRIX_SHAPES],
        line_arg="backend",
        line_vals=BACKENDS.values(),
        line_names=list(BACKENDS.keys()),
        args=dict(device=get_device(), dtype=torch.float16),
        plot_name="matmul-tflops-comparison",
        xlabel="Matrix Shape",
        ylabel="TFLOPS",
    )
]


@triton.testing.perf_report(configs)
def benchmark(
    backend: MatrixBackend,
    M: int,
    N: int,
    K: int,
    device: str,
    dtype: torch.dtype,
) -> float:
    """Benchmark function for matrix multiplication."""

    a = backend.generate_matrix(
        rows=M,
        cols=K,
        device=device,
        dtype=dtype,
    )
    b = backend.generate_matrix(
        rows=K,
        cols=N,
        device=device,
        dtype=dtype,
    )

    time_ms = triton.testing.do_bench(
        fn=lambda: backend.multiply_matrices(a, b),
        warmup=25,
        rep=100,
        grad_to_none=None,
        quantiles=None,
        fast_flush=True,
        return_mode="median",
        device_type=device,
    )

    tflops = 2 * M * N * K * 1e-12 / (time_ms * 1e-3)

    return tflops


def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Some backends may not work.")

    benchmark.run(
        print_data=True,
        diff_col=len(BACKENDS) == 2,
        show_plots=False,
        save_path="results",
    )

    results_df = pd.read_csv("results/matmul-tflops-comparison.csv")
    fig = create_3d_plot(results_df)
    fig.show()
    fig.write_html("results/matmul-tflops-comparison-3d.html")


if __name__ == "__main__":
    main()
