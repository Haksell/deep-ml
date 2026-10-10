from math import sqrt
import torch

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient (float)
		- direction: Unit vector (torch.Tensor) in direction of steepest ascent
		- descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
	"""
	gradient = torch.tensor(gradient)
	magnitude = torch.linalg.norm(gradient)
	return {
		"magnitude": magnitude,
		"direction": gradient / magnitude if magnitude else gradient,
		"descent_direction": -gradient / magnitude if magnitude else gradient,
	}