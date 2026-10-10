from collections import Counter
from math import sqrt
import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    data = sorted(data)
    n = len(data)
    mean = sum(data) / len(data)
    mode = Counter(data).most_common()[0][0]
    variance = sum((x-mean)**2 for x in data) / n
    std = sqrt(variance)
    p25 = data[n//4]
    p50 = data[n//2] if n & 1 else (data[n//2] + data[n//2-1]) / 2
    p75 = data[3*n//4]
    iqr = p75 - p25
    return {
        "mean": mean,
        "median": p50,
        "mode": mode,
        "variance": variance,
        "standard_deviation": std,
        "25th_percentile": p25,
        "50th_percentile": p50,
        "75th_percentile": p75,
        "interquartile_range": iqr,
    }