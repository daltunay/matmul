import typing as tp

import numpy as np
from numpy.typing import DTypeLike, NDArray

from .base import MatrixBackend


class NumpyBackend(MatrixBackend[NDArray]):
    @staticmethod
    def generate_matrix(
        rows: int, cols: int, dtype: DTypeLike = None, *_, **__
    ) -> NDArray:
        return (
            np.random.randn(rows, cols).astype(dtype)
            if dtype is not None
            else np.random.randn(rows, cols)
        )

    @staticmethod
    def multiply_matrices(a: NDArray, b: NDArray) -> NDArray:
        return np.matmul(a, b)

    @staticmethod
    def convert_dtype(
        dtype_str: tp.Literal["fp8", "fp16", "fp32", "fp64"]
    ) -> tp.Any | tp.NoReturn:
        dtype = {
            "fp8": None,
            "fp16": np.float16,
            "fp32": np.float32,
            "fp64": np.float64,
        }[dtype_str]

        if not dtype:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        return dtype
