import typing as tp

T = tp.TypeVar("T")


class MatrixBackend(tp.Generic[T]):
    @staticmethod
    def generate_matrix(
        rows: int, cols: int, dtype: tp.Any = None, *args, **kwargs
    ) -> T:
        raise NotImplementedError

    @staticmethod
    def multiply_matrices(a: T, b: T) -> T:
        raise NotImplementedError

    @staticmethod
    def convert_dtype(
        dtype_str: tp.Literal["fp8", "fp16", "fp32", "fp64"]
    ) -> tp.Any | tp.NoReturn:
        raise NotImplementedError
