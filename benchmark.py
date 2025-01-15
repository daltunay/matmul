import random
import typing as tp
import warnings
from math import log2

import pandas as pd
import structlog
import torch
import triton
import triton.testing

from implementations import BACKENDS, MatrixBackend
from implementations.base import DType, DTypeT
from plot import plot_results
from processing import process_results

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="triton.testing",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
)

log = structlog.get_logger()


def generate_random_shapes(
    num_shapes: int = 1000, max_dim: int = 2**14, powers_of_two: bool = True
) -> list[tuple[int, int, int]]:
    shapes = set()
    if powers_of_two:
        max_power_of_two = int(log2(max_dim))
        while len(shapes) < num_shapes:
            m = 2 ** random.randint(0, max_power_of_two)
            n = 2 ** random.randint(0, max_power_of_two)
            k = 2 ** random.randint(0, max_power_of_two)
            shapes.add((min(m, k), n, max(m, k)))
    else:
        while len(shapes) < num_shapes:
            m = random.randint(1, max_dim)
            n = random.randint(1, max_dim)
            k = random.randint(1, max_dim)
            shapes.add((min(m, k), n, max(m, k)))
    return sorted(list(shapes))


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    return "CPU"


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

    if isinstance(dtype, str):
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
        try:
            return backend.multiply_matrices(a, b)
        except Exception as e:
            logger.error("Error occurred during matmul", error=str(e))
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

    results_raw: pd.DataFrame = benchmark.run(return_df=True)[0]
    results_raw.to_csv("results_raw/matmul-tflops-comparison-raw.csv")
    print("Matrix Multiplication TFLOPS results_raw (raw):", results_raw)

    results_processed = process_results(results_raw)
    results_processed.to_csv("results_raw/matmul-tflops-comparison-processed.csv")
    print("Matrix Multiplication TFLOPS results_raw (processed):", results_processed)

    fig = plot_results(results_processed)
    fig.savefig("results_raw/matmul-tflops-comparison.png")
    fig.show()


if __name__ == "__main__":
    main()
