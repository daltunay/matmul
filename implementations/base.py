import typing as tp

from typing_extensions import Protocol

DType = tp.Literal["fp8", "fp16", "fp32", "fp64"]
T = tp.TypeVar("T", covariant=True)
DTypeT = tp.TypeVar("DTypeT")


class MatrixBackend(Protocol[T]):
    """Protocol defining the interface for matrix multiplication backends."""

    DTYPE_MAP: tp.ClassVar[dict[DType, tp.Any]] = {
        "fp8": None,
        "fp16": None,
        "fp32": None,
        "fp64": None,
    }

    @staticmethod
    def generate_matrix(
        rows: int,
        cols: int,
        dtype: DTypeT,
        device: str,
        *args: tp.Any,
        **kwargs: tp.Any,
    ) -> T:
        """Generate a matrix of given shape and type.

        Args:
            rows: Number of rows
            cols: Number of columns
            dtype: Data type for the matrix
            device: Device to create the matrix on

        Returns:
            A matrix of type T with the specified shape
        """
        raise NotImplementedError

    @staticmethod
    def multiply_matrices(a: T, b: T) -> T:
        """Multiply two matrices.

        Args:
            a: First matrix
            b: Second matrix

        Returns:
            Result of matrix multiplication
        """
        raise NotImplementedError

    @classmethod
    def convert_dtype(cls, dtype_str: DType) -> DTypeT | tp.NoReturn:
        """Convert string dtype to backend-specific dtype.

        Args:
            dtype_str: String representation of dtype

        Returns:
            Backend-specific dtype object

        Raises:
            ValueError: If dtype is not supported
        """
        dtype = cls.DTYPE_MAP[dtype_str]
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype
