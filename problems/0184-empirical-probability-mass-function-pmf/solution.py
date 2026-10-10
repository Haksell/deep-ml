from collections import Counter

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    n = len(samples)
    return sorted((k, v / n) for k, v in sorted(Counter(samples).items()))