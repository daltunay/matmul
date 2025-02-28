import typing as tp
import numpy as np
from numpy.typing import NDArray

from .base import DType, MatrixBackend


class NumpyBackend(MatrixBackend[NDArray]):
    DTYPE_MAP = {
        "fp8": None,
        "fp16": np.float16,
        "fp32": np.float32,
        "fp64": np.float64,
    }

    @staticmethod
    def generate_matrix(
        rows: int,
        cols: int,
        dtype: np.dtype,
        device: str,
        *_: tp.Any,
        **__: tp.Any
    ) -> NDArray:
        return np.random.randn(rows, cols).astype(dtype)

    @staticmethod
    def multiply_matrices(a: NDArray, b: NDArray) -> NDArray:
        return np.matmul(a, b)

    @classmethod
    def convert_dtype(cls, dtype_str: DType) -> np.dtype | tp.NoReturn:
        dtype = cls.DTYPE_MAP[dtype_str]

        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        return dtype
