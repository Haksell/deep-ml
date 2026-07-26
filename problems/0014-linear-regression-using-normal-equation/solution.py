import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X = np.array(X)
	y = np.array(y)
	res = np.linalg.inv(X.T @ X) @ X.T @ y
	res = np.round(res, 4)
	return res