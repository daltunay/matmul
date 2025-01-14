import json
import typing as tp

import structlog
import torch
import triton
import triton.testing
import pandas as pd
from implementations import BACKENDS, MatrixBackend

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
    dtype_str: str,
    warmup: int = 1,
    rep: int = 5,
) -> float | None:
    """Benchmark function for matrix multiplication."""
    log = logger.bind(backend=backend.__name__, shape=(M, N, K), dtype_str=dtype_str)
    log.info("Starting benchmark")

    try:
        dtype = backend.convert_dtype(dtype_str)
        log = log.bind(dtype=dtype)
    except ValueError:
        log.warning("dtype not supported", status="skipped")
        return None

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

    log.debug("Running benchmark")

    def safe_matmul() -> tp.Any | None:
        try:
            return backend.multiply_matrices(a, b)
        except Exception as e:
            log.error("Error occurred during matmul", error=e)
            return None
    
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
        print_data=True,
        diff_col=len(BACKENDS) == 2,
        show_plots=False,
        save_path="results",
        return_df=True,
    )


if __name__ == "__main__":
    main()
