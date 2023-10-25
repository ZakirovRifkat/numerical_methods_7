import numpy as np
from tabulate import tabulate

np.set_printoptions(formatter={"all": lambda x: f"{x:.4f}"})  # Set the format
EPS = 0.001


def tridiagonal_matrix_algorithm(a, b, c, d):
    n = len(d)
    c_new = np.zeros(n - 1)
    d_new = np.zeros(n)

    # Прямой ход (прогонка вперед)
    c_new[0] = c[0] / b[0]
    d_new[0] = d[0] / b[0]

    for i in range(1, n - 1):
        temp = b[i] - a[i] * c_new[i - 1]
        c_new[i] = c[i] / temp
        d_new[i] = (d[i] - a[i] * d_new[i - 1]) / temp

    # Обратный ход (прогонка назад)
    x = np.zeros(n)
    x[-1] = d_new[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_new[i] - c_new[i] * x[i + 1]

    return x


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


def normOfMatrix(matrix):
    max_abs_element = None

    for i in range(1, len(matrix) - 1):
        for j in range(1, len(matrix[i]) - 1):
            abs_element = abs(matrix[i][j])
            if max_abs_element is None or abs_element > max_abs_element:
                max_abs_element = abs_element

    return max_abs_element


def exactSolition(x_i, y_i):
    exact = np.zeros((len(x_i), len(y_i)))
    for i in range(len(x_i)):
        for j in range(len(y_i)):
            exact[i][j] = mu(x_i[i], y_i[j])
    return exact


def f_matrix(x_i, y_i):
    matrix = np.zeros((len(x_i), len(y_i)))
    for i in range(len(x_i)):
        for j in range(len(y_i)):
            matrix[i][j] = f(x_i[i], y_i[j])
    return matrix


def terms_1(x, y, j, u, tau, hx, hy):
    term1 = []
    term2 = []
    term3 = []
    term4 = []
    for i in range(len(x)):
        term1.append((-1 * p(x[i] - hx / 2, y[j])) / (hx**2))
        term2.append(
            (
                2 / tau
                + p(x[i] + hx / 2, y[j]) / hx**2
                + p(x[i] - hx / 2, y[j]) / hx**2
            )
        )
        term3.append((-1 * p(x[i] + hx / 2, y[j])) / (hx**2))
        g = (
            (2 * u[i][j]) / tau
            + (q(x[i], y[j] + hy / 2) * (u[i][j + 1] - u[i][j])) / hy**2
            - (q(x[i], y[j] - hy / 2) * (u[i][j] - u[i][j - 1])) / hy**2
            + f(x[i], y[i])
        )
        term4.append(g)
    return term1, term2, term3, term4


def terms_2(x, y, j, u, tau, hx, hy):
    term1 = []
    term2 = []
    term3 = []
    term4 = []
    for i in range(len(x)):
        term1.append((-1 * q(x[i], y[j] - hy / 2)) / hy**2)
        term2.append(
            (
                2 / tau
                + q(x[i], y[j] + hy / 2) / hy**2
                + q(x[i], y[j] - hy / 2) / hy**2
            )
        )
        term3.append((-1 * p(x[i], y[j] + hy / 2)) / (hy**2))
        g = (
            (2 * u[i][j]) / tau
            + (p(x[i] + hx / 2, y[j]) * (u[i + 1][j] - u[i][j])) / hx**2
            - (p(x[i] - hx / 2, y[j]) * (u[i][j] - u[i - 1][j])) / hx**2
            + f(x[i], y[i])
        )
        term4.append(g)
    return term1, term2, term3, term4


def alternatingDirectionMethod(x, y, hx, hy, tau):
    k = 0
    previous_U = fillBorderMatrix(x, y)
    current_U = fillBorderMatrix(x, y)
    U_0 = np.copy(previous_U)
    arrayOfU_K = [U_0]
    exact = exactSolition(x, y)
    # while (normOfMatrix(current_U - exact) / normOfMatrix(U_0 - exact) > EPS): #k<32
    #     print(1)
    u_half = np.zeros((len(y), len(x)))
    for i in range(1, len(x) - 1):
        # if j == 0:
        #     for i in range(len(x)):
        #         u_half[j][i] = mu(x[i], 0)
        # elif j == len(y):
        #     for i in range(len(x)):
        #         u_half[j][i] = mu(x[i], np.pi)
        # else:
        term1, term2, term3, term4 = terms_1(x, y, i, previous_U, tau, hx, hy)
        u_half[j] = tridiagonal_matrix_algorithm(term1, term2, term3, term4)
    print(u_half)
    return 0


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

sigma_1 = c1 * (4 / (step_x**2)) * ((np.sin((np.pi * step_x) / 2 * 1)) ** 2)
sigma_2 = c2 * (4 / step_x**2) * ((np.cos((np.pi * step_x) / 2 * 1)) ** 2)

delta_1 = c2 * (4 / (step_y**2)) * ((np.sin((np.pi * step_y) / (2 * np.pi))) ** 2)
delta_2 = c2 * (4 / (step_y**2)) * ((np.cos((np.pi * step_y) / (2 * np.pi))) ** 2)

sigma = max([sigma_1, sigma_2])
delta = max([delta_1, delta_2])

tauOpt = 2 / (np.sqrt(sigma * delta))
alternatingDirectionMethod(x_i, y_i, step_x, step_y, tauOpt)
