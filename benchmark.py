import torch
import triton
import triton.testing

from implementations import BACKENDS, MatrixBackend


def get_device():
    """Get the appropriate device for benchmarking."""
    if torch.cuda.is_available():
        torch.cuda.init()
        return "cuda"
    print("WARNING: CUDA not available, falling back to CPU")
    return "cpu"


# https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/#general-matrix-multiply-gemm-performance
MATRIX_SPECS = [
    {
        "M": 16384,
        "N": 8192,
        "K": 1280,
        "note": "Fused QKV Projection GEMM shape",
    },
    {
        "M": 16384,
        "N": 1024,
        "K": 8192,
        "note": "Attention Output Projection shape",
    },
    {
        "M": 16384,
        "N": 8192,
        "K": 7168,
        "note": "FFN GEMM shape",
    },
    {
        "M": 16384,
        "N": 3584,
        "K": 8192,
        "note": "FFN GEMM shape",
    },
    {
        "M": 8192,
        "N": 8192,
        "K": 8192,
        "note": "Standard GEMM shape for benchmarking",
    },
]

configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K", "dtype"],
        x_vals=[(c["M"], c["N"], c["K"]) for c in MATRIX_SPECS],
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


if __name__ == "__main__":
    main()
