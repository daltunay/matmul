import os

os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.testing

from implementations import BACKENDS, MatrixBackend


def get_device():
    """Get the appropriate device for benchmarking."""
    if torch.cuda.is_available():
        # Initialize CUDA device
        torch.cuda.init()
        return "cuda"
    print("WARNING: CUDA not available, falling back to CPU")
    return "cpu"


def get_backend_name(backend_cls) -> str:
    """Get a display name for the backend."""
    return backend_cls.__name__.replace("Backend", "").lower()


configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Arguments for x-axis
        x_vals=[
            [128, 128, 128],  # You can modify these test sizes
            [256, 256, 256],
            [512, 512, 512],
            # Add more size combinations as needed
        ],
        line_arg="backend",  # Argument name for different lines
        line_vals=[get_backend_name(backend) for backend in BACKENDS.values()],
        line_names=[get_backend_name(backend) for backend in BACKENDS.values()],
        ylabel="TFLOPS",  # Label for y-axis
        plot_name="matmul-performance-comparison",  # Plot name
        args={},
    )
]


@triton.testing.perf_report(configs)
def benchmark(M: int, N: int, K: int, backend: str):
    """Benchmark function for matrix multiplication."""
    device = get_device()

    # Get the backend class
    backend_cls: MatrixBackend = BACKENDS[backend]

    # Initialize matrices using the backend's method
    a = backend_cls.generate_matrix(
        rows=M,
        cols=K,
        device=device,
        dtype=torch.float16,
    )
    b = backend_cls.generate_matrix(
        rows=K,
        cols=N,
        device=device,
        dtype=torch.float16,
    )

    # Warm up
    backend_cls.multiply_matrices(a, b)

    # Benchmark with multiple iterations
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: backend_cls.multiply_matrices(a, b), quantiles=quantiles
    )

    # Calculate TFLOPS (2 operations per multiply-add)
    def perf(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Some backends may not work.")
    print("Running matrix multiplication benchmarks...")
    benchmark.run(print_data=True, show_plots=True)


if __name__ == "__main__":
    main()
