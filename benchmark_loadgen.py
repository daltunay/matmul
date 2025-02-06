import argparse
import json
import os
from typing import Any

import mlperf_loadgen as lg
import structlog
import torch

from benchmark_triton import generate_random_shapes
from implementations import BACKENDS, MatrixBackend
from implementations.base import DType
from utils import get_hardware_info, get_software_info

log = structlog.get_logger()


class MatrixQSL:
    """Query Sample Library for matrix multiplication benchmarks."""

    def __init__(
        self,
        samples: int,
        M: int,
        N: int,
        K: int,
        dtype: DType,
        backend: MatrixBackend,
        device: str = "cuda",
    ):
        self.samples = samples
        self.samples = samples
        self.M = M
        self.N = N
        self.K = K
        self.device = device
        self.backend = backend
        self.dtype = backend.convert_dtype(dtype)

        # Generate all matrix pairs upfront
        self.matrix_pairs = [
            (
                backend.generate_matrix(M, K, self.dtype, device),
                backend.generate_matrix(K, N, self.dtype, device),
            )
            for _ in range(samples)
        ]

        self.loaded_samples = {}

    def load_query_samples(self, sample_indices: list[int]) -> None:
        """LoadGen callback to load samples into memory."""
        for idx in sample_indices:
            self.loaded_samples[idx] = self.matrix_pairs[idx]

    def unload_query_samples(self, sample_indices: list[int]) -> None:
        """LoadGen callback to unload samples from memory."""
        for idx in sample_indices:
            if idx in self.loaded_samples:
                del self.loaded_samples[idx]

    def get_matrices(self, index: int) -> tuple[Any, Any]:
        """Get matrix pair for a given index."""
        return self.loaded_samples.get(index, (None, None))


class MatrixSUT:
    """System Under Test for matrix multiplication benchmarks."""

    def __init__(self, qsl: MatrixQSL):
        self.qsl = qsl

    def issue_queries(self, query_samples: list[lg.QuerySample]) -> None:
        """Process a batch of matrix multiplication queries."""
        for sample in query_samples:
            a, b = self.qsl.get_matrices(sample.index)
            if a is not None and b is not None:
                # Perform matrix multiplication using the specified backend
                c = self.qsl.backend.multiply_matrices(a, b)

                # Convert result to bytes for LoadGen response
                response_data = (
                    c.cpu().numpy().tobytes() if hasattr(c, "cpu") else c.tobytes()
                )

                response = lg.QuerySampleResponse(
                    sample.id, id(response_data), len(response_data)
                )
                lg.QuerySamplesComplete([response])
            else:
                log.error("Missing matrices for sample", index=sample.index)

    def flush_queries(self) -> None:
        """Flush any outstanding queries."""
        pass


def create_sut(sut: MatrixSUT) -> lg.ConstructSUT:
    """Create LoadGen SUT wrapper."""
    return lg.ConstructSUT(sut.issue_queries, sut.flush_queries)


def create_qsl(qsl: MatrixQSL) -> lg.ConstructQSL:
    """Create LoadGen QSL wrapper."""
    return lg.ConstructQSL(
        qsl.samples,
        qsl.samples,
        qsl.load_query_samples,
        qsl.unload_query_samples,
    )


def run_benchmark(
    backend: MatrixBackend,
    M: int,
    N: int,
    K: int,
    dtype: DType,
    device: str,
    target_qps: float,
    samples: int,
    log_dir: str = "./results/loadgen/",
) -> None:
    """Run LoadGen benchmark for a specific configuration."""
    logger = log.bind(
        backend=backend.__name__,
        shape=(M, N, K),
        dtype=dtype,
        device=device,
        target_qps=target_qps,
    )
    logger.info("Starting benchmark case")

    # Initialize QSL and SUT
    qsl = MatrixQSL(
        samples=samples,
        M=M,
        N=N,
        K=K,
        dtype=dtype,
        backend=backend,
        device=device,
    )
    sut_instance = MatrixSUT(qsl)

    # Create LoadGen wrappers
    sut = create_sut(sut_instance)
    qsl_wrapped = create_qsl(qsl)

    # Configure LoadGen settings for offline scenario
    settings = lg.TestSettings()
    settings.scenario = lg.TestScenario.Offline
    settings.mode = lg.TestMode.PerformanceOnly
    settings.offline_expected_qps = target_qps

    # Configure logging
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_settings = lg.LogSettings()
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_dir
    log_output_settings.copy_summary_to_stdout = False
    log_settings.enable_trace = False
    log_settings.log_output = log_output_settings

    # Run benchmark
    lg.StartTestWithLogSettings(sut, qsl_wrapped, settings, log_settings)

    # Cleanup
    lg.DestroySUT(sut)
    lg.DestroyQSL(qsl_wrapped)

    logger.info("Benchmark case completed")


def main(
    num_shapes: int,
    max_dim: int,
    powers_of_two: bool,
    target_qps: float,
    samples: int,
    output_path: str | None = None,
    direct_dims: tuple[int, int, int] | None = None,
) -> None:
    """Main benchmark runner."""
    hardware_info = get_hardware_info()
    software_info = get_software_info()

    device = hardware_info["device"]
    if not torch.cuda.is_available():
        log.warning("CUDA not available - some backends may be limited")

    # Get matrix shapes to test
    if direct_dims:
        shapes = [direct_dims]
    else:
        shapes = generate_random_shapes(
            num_shapes=num_shapes,
            max_dim=max_dim,
            powers_of_two=powers_of_two,
        )

    # Run benchmarks
    results = []
    for backend in BACKENDS:
        for M, N, K in shapes:
            for dtype in backend.value.DTYPE_MAP.keys():
                try:
                    run_benchmark(
                        backend=backend.value,
                        M=M,
                        N=N,
                        K=K,
                        dtype=dtype,
                        device=device,
                        target_qps=target_qps,
                        samples=samples,
                    )
                except Exception as e:
                    log.error(
                        "Benchmark failed",
                        backend=backend.name,
                        shape=(M, N, K),
                        dtype=dtype,
                        error=str(e),
                    )

    if output_path:
        benchmark_data = {
            "hardware": hardware_info,
            "software": software_info,
            "benchmarks": results,
        }
        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(benchmark_data, f, indent=2)
        log.info("Results saved", path=json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matrix multiplication loadgen benchmark"
    )

    # Shape configuration
    parser.add_argument(
        "--shape-mode",
        choices=["random", "direct"],
        default="random",
        help="Mode for matrix shapes",
    )
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=10,
        help="[Random mode] Number of shapes to test",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2**5,
        help="[Random mode] Maximum matrix dimension",
    )
    parser.add_argument(
        "--powers-of-two",
        action="store_true",
        help="[Random mode] Use power-of-2 dimensions",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="[Direct mode] Matrix dimensions",
    )

    # LoadGen configuration
    parser.add_argument(
        "--target-qps",
        type=float,
        default=1000,
        help="Target queries per second",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples",
    )

    args = parser.parse_args()

    if args.shape_mode == "direct" and args.shape is None:
        parser.error("Direct mode requires --shape M N K")
    if args.shape_mode == "random" and args.shape is not None:
        parser.error("Random mode does not accept --shape argument")

    return args


if __name__ == "__main__":
    args = parse_args()

    direct_dims = tuple(args.shape) if args.shape_mode == "direct" else None

    main(
        num_shapes=args.num_shapes if args.shape_mode == "random" else 1,
        max_dim=args.max_dim,
        powers_of_two=args.powers_of_two,
        target_qps=args.target_qps,
        samples=args.samples,
        output_path="./results/loadgen/matmul-benchmark.json",
        direct_dims=direct_dims,
    )
