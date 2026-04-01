import torch


def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute the inverse of a 2x2 matrix using PyTorch.

    Args:
        matrix: A 2x2 matrix (can be list, numpy array, or torch.Tensor)

    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    (a, b), (c, d) = matrix
    det = a * d - b * c
    if det == 0:
        return None
    return torch.tensor(
        [
            [d / det, -b / det],
            [-c / det, a / det],
        ]
    )