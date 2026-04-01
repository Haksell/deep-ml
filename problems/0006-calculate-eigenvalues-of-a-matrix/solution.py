from math import sqrt
import torch


def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    (a, b), (c, d) = matrix
    # Characteristic polynomial:
    # λ² - (a+d)λ + (ad-bc)
    p2 = 1
    p1 = -(a + d)
    p0 = a * d - b * c
    discriminant = p1 * p1 - 4 * p0 * p2
    assert discriminant >= 0, "subject unspecified for imaginary roots"
    sqrtd = sqrt(discriminant)
    x1 = (-p1 + sqrtd) / (2 * p2)
    x2 = (-p1 - sqrtd) / (2 * p2)
    return torch.tensor([x1, x2])