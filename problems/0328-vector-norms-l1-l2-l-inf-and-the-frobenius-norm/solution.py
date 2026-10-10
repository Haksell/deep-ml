from math import sqrt
import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    elems = arr.view(-1)
    if norm_type == "l1":
        return float(elems.abs().sum())
    elif norm_type == "l2":
        return sqrt((elems**2).sum())
    elif norm_type == "linf":
        return float(elems.abs().max())
    elif norm_type == 'frobenius':
        if arr.ndim != 2:
            raise ValueError
        return sqrt((elems**2).sum())
    else:
        raise ValueError
