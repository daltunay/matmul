import random
import typing as tp
import warnings

import pandas as pd
import structlog
import torch
import triton
import triton.testing

from implementations import BACKENDS, MatrixBackend
from implementations.base import DType, DTypeT
from plot import plot_results

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="triton.testing",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
)

log = structlog.get_logger()


def generate_random_shapes(
    num_shapes: int = 100, max_dim: int = 2**14
) -> list[tuple[int, int, int]]:
    """Generate random matrix shapes."""
    shapes = []
    for _ in range(num_shapes):
        M = random.randint(1, max_dim)
        N = random.randint(1, max_dim)
        K = random.randint(1, max_dim)
        shapes.append((M, N, K))
    return shapes


def get_device():
    """Get the appropriate device for benchmarking."""
    if torch.cuda.is_available():
        torch.cuda.init()
        return "cuda"
    print("WARNING: CUDA not available, falling back to CPU")
    return "cpu"


MATRIX_SHAPES = generate_random_shapes()

configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K", "dtype"],
        x_vals=[
            (M, N, K, dtype_str)
            for (M, N, K) in MATRIX_SHAPES
            for dtype_str in ["fp8", "fp16", "fp32", "fp64"]
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
    dtype: DTypeT | DType,
    warmup: int = 1,
    rep: int = 10,
) -> float | None:
    """Benchmark function for matrix multiplication."""
    logger = log.bind(backend=backend.__name__, shape=(M, N, K), dtype_str=dtype)
    logger.info("Starting benchmark")

    if isinstance(dtype, DType):
        try:
            dtype = backend.convert_dtype(dtype)
        except ValueError:
            logger.warning("dtype not supported", status="skipped")
            return None

    logger = logger.bind(dtype=dtype)

    logger.debug("Generating matrices")
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
        # TODO: add timeout
        try:
            return backend.multiply_matrices(a, b)
        except Exception as e:
            logger.error("Error occurred during matmul", error=e)
            return None

    logger.debug("Running benchmark")

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

    logger.info("Benchmark complete", time_ms=time_ms, tflops=tflops, status="success")

    return tflops


def main():
    if not torch.cuda.is_available():
        log.warning("cuda not available", message="Some backends may not work")

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

    results_df = results_df.pivot(
        index="shape",
        columns=["dtype"],
        values=list(BACKENDS.keys()),
    )

    print("Matrix Multiplication TFLOPS results:")
    print(results_df)

    fig = plot_results(results_df)
    fig.savefig("results/matmul-tflops-comparison.png", bbox_inches="tight", dpi=300)
    fig.show()


if __name__ == "__main__":
    main()
