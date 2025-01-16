import argparse
import glob
import os

import graphviz
import pandas as pd
import structlog
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

log = structlog.get_logger()


def load_and_prepare_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path, index_col=0)

    features = df[["M", "N", "K"]].copy()
    features["dtype_bits"] = df["dtype"].apply(lambda x: int(x[2:]))

    backend_columns = [
        col for col in df.columns if col not in ["device", "M", "N", "K", "dtype"]
    ]
    log.info("Detected backends", backends=backend_columns)

    valid_rows = df[backend_columns].notna().all(axis=1)
    features = features[valid_rows]

    performance_data = df[backend_columns][valid_rows]
    best_backend = performance_data.idxmax(axis=1)

    X, y = features, best_backend
    return X, y


def train_model(
    features: pd.DataFrame, labels: pd.Series, max_depth: int | None = None
) -> tuple[DecisionTreeClassifier, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }
    return model, metrics


def visualize_tree(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    output_file: str,
):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render(output_file, view=False, format="png", cleanup=True)


def process_single_file(csv_path: str, max_depth: int | None = None):
    log.info("Processing file", csv_path=csv_path)
    features, labels = load_and_prepare_data(csv_path)
    unique_backends = labels.unique()
    log.info("Found backends", backends=list(unique_backends))

    log.info("Training model...")
    model, metrics = train_model(features, labels, max_depth)
    log.info("Model training completed", accuracy=metrics["accuracy"])
    log.info("Classification Report:\n" + metrics["classification_report"])

    feature_names = ["M", "N", "K", "dtype_bits"]
    class_names = list(unique_backends)

    tree_path = csv_path.replace("benchmark", "decision-tree").rsplit(".", 1)[0]
    log.info("Generating decision tree visualization...")
    visualize_tree(model, feature_names, class_names, tree_path)
    log.info("Decision tree visualization saved", output_path=f"{tree_path}.png")

    return metrics


def main(max_depth: int | None = None):
    results_dir = "results"
    csv_pattern = os.path.join(results_dir, "matmul-tflops-benchmark-*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        log.error("No CSV files found", pattern=csv_pattern)
        return

    log.info("Found CSV files", files=csv_files)

    for csv_path in csv_files:
        process_single_file(csv_path, max_depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate matrix multiplication backend predictor"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of the decision tree (default: None, unlimited)",
    )
    args = parser.parse_args()
    main(max_depth=args.max_depth)
