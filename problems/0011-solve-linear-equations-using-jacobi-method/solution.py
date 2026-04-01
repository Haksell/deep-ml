import torch

def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A = torch.as_tensor(A, dtype=torch.float)
    b = torch.as_tensor(b, dtype=torch.float)
    D = A.diag()
    R = A - D.diag()
    D_inv = 1 / D
    x = torch.zeros(A.size(1))
    for _ in range(n):
        x = D_inv * (b - R @ x)
    return x