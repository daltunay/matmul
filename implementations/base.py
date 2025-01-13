import typing as tp

T = tp.TypeVar("T")


class MatrixBackend(tp.Generic[T]):
    @staticmethod
    def generate_matrix(rows: int, cols: int, inner: int) -> tuple[T, T]:
        raise NotImplementedError

    @staticmethod
    def multiply_matrices(a: T, b: T) -> T:
        raise NotImplementedError
