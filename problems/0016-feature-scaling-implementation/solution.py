import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	mini = np.min(data, axis=0)
	maxi = np.max(data, axis=0)
	standardized = (data - mini) / (maxi - mini)
	normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
	return normalized, standardized