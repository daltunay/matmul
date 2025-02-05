import typing as tp

import torch

from .base import DType, MatrixBackend


class TorchBackend(MatrixBackend[torch.Tensor]):
    DTYPE_MAP = {
        "fp8": None,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }

    @staticmethod
    def generate_matrix(
        rows: int,
        cols: int,
        dtype: torch.dtype,
        device: str = "cpu",
        *_: tp.Any,
        **__: tp.Any,
    ) -> torch.Tensor:
        return torch.randn(rows, cols, dtype=dtype, device=device)

    @staticmethod
    def multiply_matrices(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

    @classmethod
    def convert_dtype(cls, dtype_str: DType) -> torch.dtype | tp.NoReturn:
        dtype = cls.DTYPE_MAP[dtype_str]

        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        return dtype
