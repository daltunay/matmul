import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import structlog
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

from predict import load_and_prepare_data

log = structlog.get_logger()


def evaluate_model(
    model: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float]:
    """Evaluate model on train and test sets."""
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    return train_acc, test_acc


def evaluate_and_save_trees(
    model: DecisionTreeClassifier,
    features: pd.DataFrame,
    labels: pd.Series,
    output_path: str,
    num_shapes: int,
    max_depth: int,
    run: int,
):
    """Save decision tree visualizations for a specific training run."""
    depth_dir = os.path.join(os.path.dirname(output_path), f"depth_{max_depth}")
    os.makedirs(depth_dir, exist_ok=True)

    tree_path = os.path.join(depth_dir, f"tree_shapes{num_shapes}_run{run}")

    from predict import visualize_tree

    visualize_tree(
        model=model,
        feature_names=list(features.columns),
        class_names=list(labels.unique()),
        output_file=tree_path,
    )


def analyze_learning_curves(
    df: pd.DataFrame,
    num_shapes_range: list[int],
    max_depth_range: list[int],
    test_size: float = 0.2,
    n_runs: int = 5,
    output_path: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Analyze learning curves using a fixed test set for consistent evaluation.
    """
    features, labels = load_and_prepare_data(df)
    results = []

    for run in range(n_runs):
        from sklearn.model_selection import train_test_split

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=42 + run,
        )

        for num_shapes in num_shapes_range:
            log.info("Evaluating sample size", num_shapes=num_shapes)

            for max_depth in max_depth_range:
                if num_shapes < len(X_train_full):
                    X_train = pd.DataFrame()
                    y_train = pd.Series(dtype=y_train_full.dtype)

                    for label in y_train_full.unique():
                        mask = y_train_full == label
                        samples_per_class = max(
                            1, num_shapes // len(y_train_full.unique())
                        )

                        X_label = X_train_full[mask]
                        y_label = y_train_full[mask]

                        if len(X_label) > samples_per_class:
                            X_sampled, y_sampled = resample(
                                X_label,
                                y_label,
                                n_samples=samples_per_class,
                                random_state=42 + run,
                            )
                            X_train = pd.concat([X_train, X_sampled])
                            y_train = pd.concat([y_train, y_sampled])
                else:
                    X_train = X_train_full
                    y_train = y_train_full

                model = DecisionTreeClassifier(
                    max_depth=max_depth, random_state=42 + run
                )
                model.fit(X_train, y_train)

                train_acc = model.score(X_train, y_train)
                test_acc = model.score(X_test, y_test)

                from sklearn.metrics import f1_score
                train_f1 = f1_score(y_train, model.predict(X_train), average='weighted')
                test_f1 = f1_score(y_test, model.predict(X_test), average='weighted')

                if output_path and run == 0:
                    evaluate_and_save_trees(
                        model=model,
                        features=X_train,
                        labels=y_train,
                        output_path=output_path,
                        num_shapes=num_shapes,
                        max_depth=max_depth,
                        run=run,
                    )

                results.append(
                    {
                        "num_shapes": num_shapes,
                        "max_depth": max_depth,
                        "run": run,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "train_f1": train_f1,
                        "test_f1": test_f1,
                        "train_size": len(X_train),
                        "test_size": len(X_test),
                    }
                )

    return pd.DataFrame(results)


def get_output_dir(benchmark_path: str) -> str:
    """Get output directory based on benchmark path."""
    base_dir = os.path.dirname(os.path.dirname(benchmark_path))
    output_dir = os.path.join(base_dir, "learning_curves")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_metric_curves(
    results: pd.DataFrame,
    metric: str,
    output_path: str,
    title: str,
):
    """Create learning curve plot for a specific metric with confidence bands."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    n_depths = len(results["max_depth"].unique())
    colors = sns.color_palette("deep", n_depths)

    final_results = (
        results.groupby(["num_shapes", "max_depth"])
        .agg({
            f"train_{metric}": ["mean", "std"], 
            f"test_{metric}": ["mean", "std"]
        })
        .reset_index()
    )

    for idx, depth in enumerate(sorted(results["max_depth"].unique())):
        mask = final_results["max_depth"] == depth
        data = final_results[mask]
        color = colors[idx]

        # Plot training curve
        ax.plot(
            data["num_shapes"],
            data[(f"train_{metric}", "mean")],
            label=f"Train (depth={depth})",
            linestyle="--",
            color=color,
        )
        ax.fill_between(
            data["num_shapes"],
            data[(f"train_{metric}", "mean")] - data[(f"train_{metric}", "std")],
            data[(f"train_{metric}", "mean")] + data[(f"train_{metric}", "std")],
            alpha=0.1,
            color=color,
        )

        # Plot test curve
        ax.plot(
            data["num_shapes"],
            data[(f"test_{metric}", "mean")],
            label=f"Test (depth={depth})",
            linestyle="-",
            marker="o",
            color=color,
        )
        ax.fill_between(
            data["num_shapes"],
            data[(f"test_{metric}", "mean")] - data[(f"test_{metric}", "std")],
            data[(f"test_{metric}", "mean")] + data[(f"test_{metric}", "std")],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Number of Training Shapes")
    ax.set_ylabel(metric.title())
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    log.info("Saved learning curves plot", path=output_path)


def plot_learning_curves(
    results: pd.DataFrame,
    output_path: str,
):
    """Create separate plots for accuracy and F1 score metrics."""
    # Plot accuracy curves
    acc_path = f"{output_path}-accuracy.png"
    plot_metric_curves(
        results=results,
        metric="acc",
        output_path=acc_path,
        title="Learning Curves - Accuracy",
    )

    # Plot F1 score curves
    f1_path = f"{output_path}-f1.png"
    plot_metric_curves(
        results=results,
        metric="f1",
        output_path=f1_path,
        title="Learning Curves - F1 Score",
    )

    # Save raw data
    results.to_csv(f"{output_path}.csv", index=False)
    log.info("Saved learning curves data", path=f"{output_path}.csv")


def parse_range(range_str: str, for_depth: bool = False) -> list[int]:
    """
    Parse a range string like '1-100' or '2,4,8,16' into a list of integers.
    When for_depth=True, only returns odd numbers for tree depth.
    """
    if "," in range_str:
        values = [int(x) for x in range_str.split(",")]
        if for_depth:
            values = [x for x in values if x % 2 == 1]
        return values
    elif "-" in range_str:
        start, end = map(int, range_str.split("-"))
        if for_depth:
            start = start + 1 if start % 2 == 0 else start
            return list(range(start, end + 1, 2))
        return list(range(start, end + 1))
    else:
        max_val = int(range_str)
        if for_depth:
            return list(range(1, max_val + 1, 2))
        return list(range(1, max_val + 1, 5))


def main(
    benchmark_path: str,
    num_shapes_range: list[int],
    max_depth_range: list[int],
    n_runs: int,
):
    output_dir = get_output_dir(benchmark_path)
    log.info("Using output directory", path=output_dir)

    trees_dir = os.path.join(output_dir, "trees")
    os.makedirs(trees_dir, exist_ok=True)

    log.info("Loading benchmark results", path=benchmark_path)
    df = pd.read_csv(benchmark_path)

    results = analyze_learning_curves(
        df=df,
        num_shapes_range=num_shapes_range,
        max_depth_range=max_depth_range,
        n_runs=n_runs,
        output_path=os.path.join(trees_dir, "tree"),
    )

    curves_path = os.path.join(output_dir, "learning-curves")
    plot_learning_curves(results, curves_path)

    log.info(
        "Analysis complete",
        output_dir=output_dir,
        plot=f"{curves_path}.png",
        data=f"{curves_path}.csv",
        trees_dir=trees_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze learning curves from benchmark results"
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        help="Path to benchmark CSV file",
        default="results/latest/benchmarks/matmul-benchmark.csv",
    )
    parser.add_argument(
        "--max-num-shapes",
        type=str,
        default="100",
        help="Maximum number of shapes to test (will create a linear range from 1)",
    )
    parser.add_argument(
        "--max-depth-range",
        type=str,
        default="2-10",
        help="Range of tree depths to test (e.g., '2-10' or '2,3,4,5')",
    )
    parser.add_argument(
        "--n-runs", type=int, default=5, help="Number of runs to average over"
    )

    args = parser.parse_args()

    main(
        benchmark_path=args.benchmark_path,
        num_shapes_range=parse_range(args.max_num_shapes),
        max_depth_range=parse_range(args.max_depth_range, for_depth=True),
        n_runs=args.n_runs,
    )
