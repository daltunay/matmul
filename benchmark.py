import argparse
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
            shapes.add((m, n, k))
    else:
        while len(shapes) < num_shapes:
            m = random.randint(1, max_dim)
            n = random.randint(1, max_dim)
            k = random.randint(1, max_dim)
            shapes.add((m, n, k))
    return sorted(list(shapes))


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    return "CPU"


def main(num_shapes: int, max_dim: int, powers_of_two: bool, warmup: int, rep: int):
    if not torch.cuda.is_available():
        log.warning("cuda not available", message="Some backends may not work")

    MATRIX_SHAPES = generate_random_shapes(
        num_shapes=num_shapes,
        max_dim=max_dim,
        powers_of_two=powers_of_two,
    )

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
            plot_name="matmul",
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
    ) -> float | None:
        """Benchmark function for matrix multiplication."""
        logger = log.bind(
            backend=backend.__name__,
            shape=(M, N, K),
            dtype_str=dtype,
            device=device,
            device_name=get_device_name(),
        )
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

        logger.info(
            "Benchmark complete", time_ms=time_ms, tflops=tflops, status="success"
        )

        return tflops

    results_df: pd.DataFrame = benchmark.run(return_df=True)[0]

    device_name = get_device_name()
    results_df.insert(0, "device", device_name)
    log.info("Matrix Multiplication TFLOPS results:\n" + results_df.to_string())

    device_name_fmt = device_name.lower().replace(" ", "-")
    results_df.to_csv(f"results/matmul-tflops-benchmark-{device_name_fmt}.csv")

    fig = plot_results(results_df)
    fig.savefig(f"results/matmul-tflops-plot-{device_name_fmt}.png")
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix multiplication benchmark")
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=1000,
        help="Number of matrix shapes to generate",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2**10,
        help="Maximum matrix dimension",
    )
    parser.add_argument(
        "--powers-of-two",
        action="store_true",
        help="Generate matrix shapes with powers of two",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations per benchmark",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=10,
        help="Number of benchmark repetitions",
    )

    args = parser.parse_args()

    main(
        num_shapes=args.num_shapes,
        max_dim=args.max_dim,
        powers_of_two=args.powers_of_two,
        warmup=args.warmup,
        rep=args.rep,
    )

# example usage:
# python benchmark.py --num-shapes 100 --max-dim 1024 --powers-of-two --warmup 1 --rep 10
