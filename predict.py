import argparse
import glob
import os

import graphviz
import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import classification_report, f1_score

log = structlog.get_logger()


def load_and_prepare_data(
    data: str | pd.DataFrame, max_samples: int | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data with error handling for different column structures."""
    if isinstance(data, str):
        df = pd.read_csv(data, index_col=0)
    else:
        df = data.copy()

    required_columns = ["M", "N", "K", "dtype", "backend", "tflop/s"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        log.error(
            "Missing required columns in CSV file",
            missing=missing_columns,
            available=list(df.columns),
        )
        raise ValueError(f"CSV file missing required columns: {missing_columns}")

    try:
        df["config_id"] = df.apply(
            lambda x: f"{x['M']}_{x['N']}_{x['K']}_{x['dtype']}", axis=1
        )

        pivot_df = df.pivot(index="config_id", columns="backend", values="tflop/s")

        features_df = df.drop_duplicates("config_id").set_index("config_id")
        features = pd.DataFrame(
            {
                "M": features_df["M"],
                "N": features_df["N"],
                "K": features_df["K"],
                "dtype_bits": features_df["dtype"].apply(lambda x: int(x[2:])),
            }
        )

        best_backend = pd.Series(index=pivot_df.index, dtype="object")
        for idx in pivot_df.index:
            row = pivot_df.loc[idx]
            valid_results = row.dropna()
            if len(valid_results) > 0:
                best_backend[idx] = valid_results.idxmax()

        valid_configs = best_backend.dropna().index
        features = features.loc[valid_configs]
        best_backend = best_backend.loc[valid_configs]

        if max_samples is not None and max_samples < len(valid_configs):
            selected_configs = np.random.choice(
                valid_configs, size=max_samples, replace=False
            )
            features = features.loc[selected_configs]
            best_backend = best_backend.loc[selected_configs]

        log.info(
            "Data preparation complete",
            num_samples=len(features),
            feature_columns=list(features.columns),
            unique_backends=list(best_backend.unique()),
            backend_counts=best_backend.value_counts().to_dict(),
            total_configs=len(pivot_df),
            valid_configs=len(valid_configs),
        )

        return features, best_backend

    except Exception as e:
        log.error(
            "Error processing data",
            error=str(e),
            columns=list(df.columns),
            first_row=df.iloc[0].to_dict() if not df.empty else None,
        )
        raise


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
        "f1": f1_score(y_test, y_pred, average="weighted"),
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
        proportion=True,
        label="all",
        impurity=False,
        precision=2,
    )

    legend = """
    /*
    Node explanation:
    - First line: test condition (e.g., "M <= 512")
    - Second line: sample distribution
      samples = X (shows how many samples reach this node)
      value = [A, B, C] (how many samples belong to each class)
    - Color indicates majority class in the node
    - Darker color means higher purity (more samples of majority class)
    */
    """

    dot_data = legend + dot_data
    graph = graphviz.Source(dot_data)
    graph.render(output_file, view=False, format="png", cleanup=True)

    log.info(
        "Tree visualization guide",
        message="Generated decision tree visualization:",
        png_file=f"{output_file}.png",
        node_colors="Colors indicate the majority backend in each node",
        color_intensity="Darker colors mean higher confidence (more samples of one backend)",
        distribution="[X,Y,Z] shows sample counts for each backend",
        decisions="Internal nodes show the decision criteria (e.g., M <= 512)",
    )


def process_single_file(csv_path: str, max_depth: int | None = None):
    log.info("Processing file", csv_path=csv_path)
    features, labels = load_and_prepare_data(csv_path)
    unique_backends = labels.unique()
    log.info("Found backends", backends=list(unique_backends))

    log.info("Training model...")
    model, metrics = train_model(features, labels, max_depth)
    log.info("Model training completed", f1_score=metrics["f1"])
    log.info("Classification Report:\n" + metrics["classification_report"])

    feature_names = ["M", "N", "K", "dtype_bits"]
    class_names = list(unique_backends)

    trees_dir = os.path.join(os.path.dirname(csv_path), "trees")
    os.makedirs(trees_dir, exist_ok=True)

    base_name = (
        os.path.basename(csv_path)
        .replace("benchmark", "decision-tree")
        .rsplit(".", 1)[0]
    )
    if max_depth is not None:
        base_name = f"{base_name}_depth{max_depth}"
    tree_path = os.path.join(trees_dir, base_name)

    log.info("Generating decision tree visualization...")
    visualize_tree(model, feature_names, class_names, tree_path)
    log.info(
        "Decision tree visualization saved",
        png_path=f"{tree_path}.png",
    )

    return metrics


def main(max_depth: int | None = None):
    results_dir = "results"
    csv_pattern = os.path.join(results_dir, "*.csv")
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
