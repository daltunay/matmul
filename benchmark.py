import argparse
import json
import os
import random
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
from utils import get_hardware_info, get_software_info

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


def main(
    num_shapes: int,
    max_dim: int,
    powers_of_two: bool,
    warmup_ms: float,
    repetition_ms: float,
    output_path: str | None = None,
    direct_dims: tuple[int, int, int] | None = None,
    regenerate_matrices: bool = False,
    num_matrices: int = 1,
) -> pd.DataFrame:
    logger = log.bind(
        shapes=num_shapes if direct_dims is None else "direct",
        max_dim=max_dim,
        powers_of_two=powers_of_two,
        warmup_ms=warmup_ms,
        repetitions=repetition_ms,
    )
    logger.info("Starting benchmark suite")

    hardware_info = get_hardware_info()
    software_info = get_software_info()

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
            line_vals=[backend.value for backend in BACKENDS],
            line_names=[backend.name for backend in BACKENDS],
            args=dict(device=hardware_info["device"]),
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
            device_name=hardware_info["name"],
        )
        logger.info("Starting benchmark case")

        try:
            dtype = backend.convert_dtype(dtype) if isinstance(dtype, str) else dtype
        except ValueError:
            logger.warning("Unsupported dtype combination", status="skipped")
            return None

        logger = logger.bind(dtype=dtype)
        logger.debug("Generating test matrices")

        total_time_ms = 0
        iterations = num_matrices if regenerate_matrices else 1
        per_matrix_ms = repetition_ms / iterations

        for i in range(iterations):
            if regenerate_matrices or i == 0:
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

            def safe_matmul() -> None:
                """Safe matrix multiplication function. Does not return anything."""
                try:
                    backend.multiply_matrices(a, b)
                except Exception as e:
                    logger.error("Error occurred during matmul", error=str(e)[:10] + "...")

            logger.debug("Running benchmark")

            time_ms = triton.testing.do_bench(
                fn=safe_matmul,
                warmup=warmup_ms,
                rep=per_matrix_ms,
                grad_to_none=None,
                quantiles=None,
                return_mode="median",
            )
            total_time_ms += time_ms

        avg_time_ms = total_time_ms / iterations
        tflops = (2 * M * N * K * 1e-12) / (avg_time_ms * 1e-3)
        logger.info(
            "Case complete",
            time_ms=f"{avg_time_ms:.4f}",
            tflops=f"{tflops:.4f}",
            matrices=iterations,
        )

        return avg_time_ms

    results: pd.DataFrame = benchmark.run(return_df=True)[0]
    print(results)
    log.info("Benchmark completed", total_cases=len(results))

    df = pd.melt(
        results,
        id_vars=["M", "N", "K", "dtype"],
        var_name="backend",
        value_name="time_ms",
    )

    df["total_ops"] = 2 * df["M"] * df["N"] * df["K"]
    df["tflop/s"] = (df["total_ops"] * 1e-12) / (df["time_ms"] * 1e-3)

    print("Benchmark results:", df)

    benchmark_data = {
        "hardware": hardware_info,
        "software": software_info,
        "benchmarks": json.loads(df.to_json(orient="records")),
    }

    if output_path:
        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(benchmark_data, f, indent=2)
        log.info("Results saved", path=json_path)

    logger.info(
        "Benchmark suite complete",
        total_cases=len(df),
        successful_cases=len(df.dropna()),
        success_rate=len(df.dropna()) / len(df),
        output_path=output_path,
    )

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matrix multiplication benchmark")

    # Shape configuration
    parser.add_argument(
        "--shape-mode",
        choices=["random", "direct"],
        default="random",
        help="Mode for matrix shapes: 'random' for multiple random shapes or 'direct' for a single shape",
    )

    # Random shape mode arguments
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=100,
        help="[Random mode] Number of matrix shapes to generate",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2**10,
        help="[Random mode] Maximum matrix dimension",
    )
    parser.add_argument(
        "--powers-of-two",
        action="store_true",
        help="[Random mode] Generate matrix shapes with powers of two",
    )

    # Direct shape mode arguments
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="[Direct mode] Matrix dimensions as 'M N K'",
    )

    # Common arguments
    parser.add_argument(
        "--warmup-ms",
        type=float,
        default=1,
        help="Warmup duration per matmul",
    )
    parser.add_argument(
        "--repetition-ms",
        type=float,
        default=10,
        help="Repeat duration per matmul",
    )

    parser.add_argument(
        "--regenerate-matrices",
        action="store_true",
        help="Generate new matrices for each test iteration",
    )
    parser.add_argument(
        "--num-matrices",
        type=int,
        default=1,
        help="Number of different matrices to test when regenerate-matrices is enabled",
    )

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.shape_mode == "direct" and args.shape is None:
        parser.error("Direct mode requires --shape M N K")
    if args.shape_mode == "random" and args.shape is not None:
        parser.error("Random mode does not accept --shape argument")

    # Validate matrix regeneration arguments
    if args.num_matrices > 1 and not args.regenerate_matrices:
        parser.error("--num-matrices can only be used with --regenerate-matrices")
    if args.num_matrices < 1:
        parser.error("--num-matrices must be at least 1")
    if args.regenerate_matrices and args.num_matrices == 1:
        parser.warning("--regenerate-matrices has no effect with --num-matrices=1")

    return args


if __name__ == "__main__":
    args = parse_args()

    direct_dims = tuple(args.shape) if args.shape_mode == "direct" else None

    main(
        num_shapes=args.num_shapes if args.shape_mode == "random" else 1,
        max_dim=args.max_dim,
        powers_of_two=args.powers_of_two,
        warmup_ms=args.warmup_ms,
        repetition_ms=args.repetition_ms,
        output_path="./results/matmul-benchmark.json",
        direct_dims=direct_dims,
        regenerate_matrices=args.regenerate_matrices,
        num_matrices=args.num_matrices,
    )
