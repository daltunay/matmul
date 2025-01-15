import pandas as pd

from implementations import BACKENDS


def process_results(df: pd.DataFrame) -> pd.DataFrame:
    df["shape"] = (
        "("
        + df["M"].astype(str)
        + ", "
        + df["N"].astype(str)
        + ", "
        + df["K"].astype(str)
        + ")"
    )

    df = df.pivot(
        index="shape",
        columns=["dtype"],
        values=list(BACKENDS.keys()),
    )

    return df
