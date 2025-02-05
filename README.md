# Matrix Multiplication Benchmarks

Benchmark and compare different matrix multiplication implementations using various backends.

## Setup

1. Install uv (Python package installer):
```bash
pip install uv
```

2. Install dependencies and activate environment:
```bash
uv sync
source .venv/bin/activate
```

## Usage

### Running Benchmarks

You can run benchmarks in two modes:

1. Random Shapes Mode:
```bash
python benchmark.py \
    --num-shapes 100 \  # Number of matrix shapes to test
    --max-dim 1024 \    # Maximum matrix dimension
    --powers-of-two \   # Use power of 2 dimensions
    --warmup_ms 1 \     # Warmup time (ms) for each matmul
    --repetition_ms 10  # Repetition time (ms) for each matmul
```

2. Direct Dimensions Mode:
```bash
python benchmark.py \
    --M 1024 \         # First matrix rows
    --N 32 \           # Second matrix columns
    --K 2048 \         # First matrix columns/Second matrix rows
    --warmup_ms 1 \    # 
    --repetition_ms 10 #
```

### Analyzing Results

Generate decision trees from benchmark results:
```bash
python predict.py --max-depth 3  # Limit tree depth for better interpretability
```

## Output Files

The benchmark generates several files in the `results/` directory:
- `matmul-tflops-benchmark-{device}.csv`: Raw benchmark data
- `matmul-tflops-plot-{device}.png`: Performance comparison plots
- `matmul-tflops-decision-tree-{device}.png`: Decision tree visualization

Where `{device}` is your GPU model (e.g., 'a100', 'v100') or 'cpu'.
