import torch
import triton
import triton.language as tl


def copy(x, bs, kernel_fn):
    z = torch.zeros_like(x)
    n = x.numel()
    n_blocks = triton.cdiv(n, bs)
    grid = (n_blocks,)

    kernel_fn[grid](x, z, n, bs)

    return z


@triton.jit
def copy_k(x_ptr, z_ptr, n, bs: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * bs + tl.arange(0, bs)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask)
    tl.store(z_ptr + offs, x, mask)


def test():
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    z = copy(x, bs=2, kernel_fn=copy_k)
    print(z)


if __name__ == "__main__":
    test()
