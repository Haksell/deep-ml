import numpy as np

def derive(coeffs: list) -> list:
    n = len(coeffs)
    return [(n - i - 1) * c for i, c in enumerate(coeffs[:-1])]

def evaluate(coeffs: list, x: float) -> float:
    res = 0
    for c in coeffs:
        res *= x
        res += c
    return res

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    g_prime_coeffs = derive(g_coeffs)
    h_prime_coeffs = derive(h_coeffs)
    g = evaluate(g_coeffs, x)
    gp = evaluate(g_prime_coeffs, x)
    h = evaluate(h_coeffs, x)
    hp = evaluate(h_prime_coeffs, x)
    return (gp*h - g*hp) / h**2
