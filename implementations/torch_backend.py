import typing as tp

import torch

from .base import MatrixBackend


class TorchBackend(MatrixBackend[torch.Tensor]):
    @staticmethod
    def generate_matrix(
        rows: int, cols: int, dtype: torch.dtype, device: str, *_, **__
    ) -> torch.Tensor:
        return torch.randn(rows, cols, dtype=dtype, device=device)

    @staticmethod
    def multiply_matrices(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

    @staticmethod
    def convert_dtype(
        dtype_str: tp.Literal["fp8", "fp16", "fp32", "fp64"]
    ) -> tp.Any | tp.NoReturn:
        dtype = {
            "fp8": None,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "fp64": torch.float64,
        }[dtype_str]

        if not dtype:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        return dtype
