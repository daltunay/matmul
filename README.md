# Matrix Multiplication Benchmarks

Benchmark suite comparing various matrix multiplication implementations across different backends.

## Features

- Multiple backend implementations (PyTorch, Triton, etc.)
- Support for different precision types (FP16, FP32, FP64)
- Comprehensive benchmarking with warmup phases
- Performance visualization and analysis tools

## Setup

```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Or using uv (recommended)
pip install uv
uv sync
source .venv/bin/activate
```

## Usage

### Running Benchmarks

The benchmark script supports two modes of operation:

#### 1. Random Shapes Mode (default)
Generate and test multiple random matrix shapes:

```bash
python benchmark.py --shape-mode random \
    --num-shapes 100 \      # Number of shapes to test
    --max-dim 1024 \        # Maximum dimension size
    --powers-of-two \       # Use power-of-2 dimensions
    --warmup-ms 10 \        # Warmup duration per matmul
    --repetition-ms 100     # Repetition duration per matmul
```

#### 2. Direct Shape Mode
Test a specific matrix multiplication shape:

```bash
python benchmark.py --shape-mode direct \
    --shape 1024 2048 32 \    # M N K dimensions
    --warmup_ms 10 \          # Warmup duration per matmul
    --repetition_ms 100       # Total repetition duration
```

By default, the benchmark uses the same matrices for all repetitions. When `--regenerate-matrices` is enabled, it will generate new matrices from a normal distribution for each test, spreading the `repetition_ms` time across all matrices. Warmup is performed before testing each new matrix. You would need to provide `--num-matrices` argument too.

Example:

```bash
python benchmark.py --shape-mode random \
    --num-shapes 100 \ to test
    --max-dim 1024 \
    --powers-of-two \
    --warmup-ms 10 \
    --repetition-ms 100 \
    --regenerate-matrices \ # Enable matrix regeneration
    --num-matrices 5        # Number of different matrices to test
```

### Output Files

The benchmark generates several files in the `results/benchmarks/` directory:
- `matmul-benchmark.json`: Raw benchmark data including:
  - Hardware information
  - Software versions
  - Benchmark results
