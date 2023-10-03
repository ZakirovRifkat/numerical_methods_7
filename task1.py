import numpy as np

EPS = 0.001

np.set_printoptions(formatter={"all": lambda x: f"{x:.4f}"})  # Set the format


def spectralRadius(xi):
    return (1 - xi) / (1 + xi)


def q(x, y):
    return 1


def p(x, y):
    return 1 + 2 * x


def f(x, y):
    return -1 * (
        2 * y**2
        + 2 * y**3
        + 8 * x * y**2
        + 8 * x * y**3
        + 2 * x**2
        + 6 * x**2 * y
    )


def mu(x, y):
    return x**2 * y**2 + x**2 * y**3


def xi(c1, c2, d1, d2, step_x, step_y):
    sigma = c1 * (4 / (step_x**2)) * ((np.sin((np.pi * step_x) / 2)) ** 2) + d1 * (
        4 / (step_y**2)
    ) * ((np.sin((np.pi * step_y) / (2 * np.pi))) ** 2)
    delta = c2 * (4 / (step_x**2)) * ((np.cos((np.pi * step_x) / 2)) ** 2) + d2 * (
        4 / (step_y**2)
    ) * ((np.cos((np.pi * step_y) / (2 * np.pi))) ** 2)

    return sigma / delta


def fillBorderMatrix(x_i, y_i):
    N = len(x_i)
    M = len(y_i)

    matrix = np.zeros((N, M))
    for i, val in enumerate(y_i):
        matrix[0][i] = mu(0, val)
        matrix[N - 1][i] = mu(x_i[N - 1], val)

    for i, val in enumerate(x_i):
        matrix[i][0] = mu(val, 0)
        matrix[i][M - 1] = mu(val, y_i[M - 1])

    return matrix


def find_max_abs_element(matrix):
    max_abs_element = None

    for i in range(1, len(matrix) - 1):
        for j in range(1, len(matrix[i]) - 1):
            abs_element = abs(matrix[i][j])
            if max_abs_element is None or abs_element > max_abs_element:
                max_abs_element = abs_element

    return max_abs_element


def simpleItteration(x, y, step_x, step_y):
    k = 0
    previous_U = fillBorderMatrix(x, y)
    current_U = fillBorderMatrix(x, y)
    exact_U = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            exact_U[i][j] = mu(x[i], y[j])

    U_0 = np.copy(previous_U)

    while (
        find_max_abs_element(current_U - exact_U) / find_max_abs_element(U_0 - exact_U)
        > EPS
    ):
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                a = (p(x[i] - step_x / 2, y[j]) * previous_U[i - 1][j]) / step_x**2
                b = (p(x[i] + step_x / 2, y[j]) * previous_U[i + 1][j]) / step_x**2
                c = (q(x[i], y[j] - step_y / 2) * previous_U[i][j - 1]) / step_y**2
                d = (q(x[i], y[j] + step_y / 2) * previous_U[i][j + 1]) / step_y**2
                a_denom = (p(x[i] - step_x / 2, y[j])) / step_x**2
                b_denom = (p(x[i] + step_x / 2, y[j])) / step_x**2
                c_denom = (q(x[i], y[j] - step_y / 2)) / step_y**2
                d_denom = (q(x[i], y[j] + step_y / 2)) / step_y**2
                current_U[i][j] = (a + b + c + d + f(x[i], y[j])) / (
                    a_denom + b_denom + c_denom + d_denom
                )
        k += 1

        previous_U = np.copy(current_U)
    return current_U, k


x_0 = 0
x_n = 1
y_0 = 0
y_m = np.pi
N = 5
M = 15
step_x = (x_n) / N
step_y = (y_m) / M
x_i = np.arange(x_0, x_n + step_x, step_x)
y_i = np.arange(y_0, y_m + step_y, step_y)

c1 = 1
c2 = 3
d1 = 1
d2 = 1

ksi = xi(c1, c2, d1, d2, step_x, step_y)
print("spectr = ", spectralRadius(ksi))
simpleItterationMatrix, amountItter = simpleItteration(x_i, y_i, step_x, step_y)
print(amountItter)
