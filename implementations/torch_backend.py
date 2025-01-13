import torch

from .base import MatrixBackend


class TorchBackend(MatrixBackend[torch.Tensor]):
    @staticmethod
    def generate_matrix(
        rows: int, cols: int, device: str, dtype: torch.dtype, *_, **__
    ) -> torch.Tensor:
        return torch.randn(rows, cols, device=device, dtype=dtype)

    @staticmethod
    def multiply_matrices(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)
