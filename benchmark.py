import json
import typing as tp
import warnings

import pandas as pd
import structlog
import torch
import triton
import triton.testing

from implementations import BACKENDS, DTypeT, MatrixBackend
from plot import plot_results

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="triton.testing",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
)

with open("matrix_specs.json", "r") as f:
    # real-world (M, N, K) shapes for Llama 70B production training
    # https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/#general-matrix-multiply-gemm-performance
    # https://github.com/pytorch-labs/float8_experimental/blob/main/benchmarks/bench_matmul.py#L58
    MATRIX_SHAPES = json.load(f)

logger = structlog.get_logger()


def get_device():
    """Get the appropriate device for benchmarking."""
    if torch.cuda.is_available():
        torch.cuda.init()
        return "cuda"
    print("WARNING: CUDA not available, falling back to CPU")
    return "cpu"


configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K", "dtype"],
        x_vals=[
            (c["M"], c["N"], c["K"], dtype_str)
            for c in MATRIX_SHAPES
            for dtype_str in [
                "fp8",
                "fp16",
                "fp32",
            ]
        ],
        line_arg="backend",
        line_vals=BACKENDS.values(),
        line_names=list(BACKENDS.keys()),
        args=dict(device=get_device()),
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
    dtype: DTypeT | str,
    warmup: int = 0,
    rep: int = 1,
) -> float | None:
    """Benchmark function for matrix multiplication."""
    log = logger.bind(backend=backend.__name__, shape=(M, N, K), dtype_str=dtype)
    log.info("Starting benchmark")

    if isinstance(dtype, str):
        try:
            dtype = backend.convert_dtype(dtype)
        except ValueError:
            log.warning("dtype not supported", dtype=dtype, status="skipped")
            return None

    log = log.bind(dtype=dtype)

    log.debug("Generating matrices")
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

    def safe_matmul() -> tp.Any | None:
        try:
            return backend.multiply_matrices(a, b)
        except Exception as e:
            log.error("Error occurred during matmul", error=e)
            return None

    log.debug("Running benchmark")

    time_ms = triton.testing.do_bench(
        fn=safe_matmul,
        warmup=warmup,
        rep=rep,
        grad_to_none=None,
        quantiles=None,
        fast_flush=True,
        return_mode="median",
        device_type=device,
    )
    tflops = 2 * M * N * K * 1e-12 / (time_ms * 1e-3)

    log.info("Benchmark complete", time_ms=time_ms, tflops=tflops, status="success")

    return tflops


def main():
    if not torch.cuda.is_available():
        logger.warning("cuda not available", message="Some backends may not work")

    results_df: pd.DataFrame = benchmark.run(
        print_data=False,
        diff_col=len(BACKENDS) == 2,
        show_plots=False,
        save_path="results",
        return_df=True,
    )[0]

    results_df["shape"] = (
        "("
        + results_df["M"].astype(str)
        + ", "
        + results_df["N"].astype(str)
        + ", "
        + results_df["K"].astype(str)
        + ")"
    )

    results_df_wide = results_df.pivot(
        index="shape", columns=["dtype"], values=list(BACKENDS.keys())
    )

    print("Matrix Multiplication TFLOPS results:")
    print(results_df_wide)

    fig = plot_results(results_df_wide)
    fig.savefig(
        "results/matmul-tflops-comparison-barplot.png",
        bbox_inches="tight",
        dpi=300,
    )
    fig.show()


if __name__ == "__main__":
    main()
