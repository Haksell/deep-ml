def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
	exp = (n + 1) / 2
	exp2 = (n + 1) * (2 * n + 1) / 6
	var = exp2 - exp**2
	return (exp, var)