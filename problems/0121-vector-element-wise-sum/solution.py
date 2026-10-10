import torch

def vector_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | int:
    return -1 if len(a) != len(b) else a + b
