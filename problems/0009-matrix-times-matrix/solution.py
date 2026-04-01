import torch


def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    a = torch.as_tensor(a, dtype=torch.float)
    b = torch.as_tensor(b, dtype=torch.float)
    return a @ b if a.size(1) == b.size(0) else torch.tensor(-1)
