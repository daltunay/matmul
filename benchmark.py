import argparse
import os
import random
import sys
import typing as tp
import warnings
from math import log2

import pandas as pd
import structlog
import torch
import torch.version
import triton
import triton.testing

from implementations import BACKENDS, MatrixBackend
from implementations.base import DType, DTypeT
from plot import plot_benchmarks

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
    random.seed(42)

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


def get_device_count() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def get_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_torch_version() -> str:
    return torch.__version__


def get_cuda_version() -> str:
    if torch.cuda.is_available():
        return torch.version.cuda
    return "N/A"


def get_triton_version() -> str:
    return triton.__version__


def main(
    num_shapes: int,
    max_dim: int,
    powers_of_two: bool,
    warmup_ms: float,
    repetition_ms: float,
    output_path: str | None = None,
    direct_dims: tuple[int, int, int] | None = None,
) -> pd.DataFrame:
    logger = log.bind(
        shapes=num_shapes if direct_dims is None else "direct",
        max_dim=max_dim,
        powers_of_two=powers_of_two,
        warmup_ms=warmup_ms,
        repetitions=repetition_ms,
    )
    logger.info("Starting benchmark suite")

    if not torch.cuda.is_available():
        log.warning("CUDA not available - some backends may be limited")

    MATRIX_SHAPES = (
        [direct_dims]
        if direct_dims is not None
        else generate_random_shapes(
            num_shapes=num_shapes,
            max_dim=max_dim,
            powers_of_two=powers_of_two,
        )
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
            dtype=dtype,
            device=device,
            device_name=get_device_name(),
        )
        logger.info("Starting benchmark case")

        try:
            dtype = backend.convert_dtype(dtype) if isinstance(dtype, str) else dtype
        except ValueError:
            logger.warning("Unsupported dtype combination", status="skipped")
            return None

        logger = logger.bind(dtype=dtype)
        logger.debug("Generating test matrices")

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
            warmup=warmup_ms,
            rep=repetition_ms,
            grad_to_none=None,
            quantiles=None,
            return_mode="median",
        )

        tflops = (2 * M * N * K * 1e-12) / (time_ms * 1e-3)
        logger.info("Case complete", time_ms=f"{time_ms:.4f}", tflops=f"{tflops:.4f}")
        return time_ms

    results: pd.DataFrame = benchmark.run(return_df=True)[0]
    log.info("Benchmark completed", total_cases=len(results))

    df = pd.melt(
        results,
        id_vars=["M", "N", "K", "dtype"],
        var_name="backend",
        value_name="time_ms",
    )

    df = df.dropna(subset=["time_ms"])

    df["total_ops"] = 2 * df["M"] * df["N"] * df["K"]

    df["tflop/s"] = (df["total_ops"] * 1e-12) / (df["time_ms"] * 1e-3)

    df["python_version"] = get_python_version()
    df["torch_version"] = get_torch_version()
    df["cuda_version"] = get_cuda_version()
    df["device_name"] = get_device_name()
    df["device_count"] = get_device_count()

    df = df[
        [
            "device_name",
            "device_count",
            "python_version",
            "torch_version",
            "cuda_version",
            "M",
            "N",
            "K",
            "dtype",
            "backend",
            "time_ms",
            "total_ops",
            "tflop/s",
        ]
    ]

    print("Benchmark results:", df)

    if output_path:
        df.to_csv(output_path)
        log.info("Results saved", path=output_path)

        plot_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]

        fig_regular, fig_normalized = plot_benchmarks(df)
        log.info("Plots generated and saved", dir=os.path.dirname(output_path))
        fig_regular.write_html(os.path.join(plot_dir, f"{base_name}-plot.html"))
        fig_normalized.write_html(
            os.path.join(plot_dir, f"{base_name}-plot-normalized.html")
        )

    logger.info(
        "Benchmark suite complete",
        total_cases=len(results),
        successful_cases=len(df),
        success_rate=f"{len(df)/len(results)*100:.4f}%",
        output_path=output_path,
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix multiplication benchmark")
    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument(
        "--num-shapes",
        type=int,
        default=1000,
        help="Number of matrix shapes to generate",
    )
    shape_group.add_argument(
        "--M",
        type=int,
        help="Direct M dimension for matrix multiplication",
    )
    parser.add_argument(
        "--N",
        type=int,
        help="Direct N dimension for matrix multiplication",
    )
    parser.add_argument(
        "--K",
        type=int,
        help="Direct K dimension for matrix multiplication",
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
        "--warmup_ms",
        type=float,
        default=1,
        help="Number of warmup_ms iterations per benchmark",
    )
    parser.add_argument(
        "--repetition_ms",
        type=float,
        default=10,
        help="Number of benchmark repetitions",
    )

    args = parser.parse_args()
    
    direct_dims = None
    if args.M is not None:
        if args.N is None or args.K is None:
            parser.error("When using direct dimensions, all --M, --N, and --K must be provided")
        direct_dims = (args.M, args.N, args.K)

    main(
        num_shapes=args.num_shapes,
        max_dim=args.max_dim,
        powers_of_two=args.powers_of_two,
        warmup_ms=args.warmup_ms,
        repetition_ms=args.repetition_ms,
        output_path="./results/benchmarks/matmul-benchmark.csv",
        direct_dims=direct_dims,
    )
