import numpy as np
import scipy.linalg as sp


def spectralRadius(matrix):
    eigenvalues = sp.eigvals(matrix)
    return max(abs(eigenvalues))


def q(x, y):
    return 1


def p(x, y):
    return 1 + 2 * x


def f(x, y):
    return (
        2 * y**2
        + 2 * y**3
        + 8 * x * y**2
        + 8 * x * y**3
        + 2 * x**2
        + 6 * x**2 * y
    )


def mu(x, y):
    return x**2 * y**2 + x**2 * y**3


if __name__ == "__main__":
    print(f(0, np.pi))
