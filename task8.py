import numpy as np
from tabulate import tabulate

np.set_printoptions(formatter={"all": lambda x: f"{x:.4f}"})  # Set the format
EPS = 0.0001


def calculate_lhu(u, x, y, hx, hy):
    lhu = np.zeros((len(x), len(y)))
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            term1 = (p(x[i] + hx / 2, y[j]) * (u[i + 1][j] - u[i][j])) / hx**2
            term2 = (p(x[i] - hx / 2, y[j]) * (u[i][j] - u[i - 1][j])) / hx**2
            term3 = (q(x[i], y[j] + hy / 2) * (u[i][j + 1] - u[i][j])) / hy**2
            term4 = (q(x[i], y[j] - hy / 2) * (u[i][j] - u[i][j - 1])) / hy**2
            lhu[i][j] = term1 - term2 + term3 - term4
    return lhu


def tridiagonal_matrix_algorithm(A, B, C, G):
    n = len(A)
    s = np.zeros(n)
    t = np.zeros(n)
    y = np.zeros(n)
    s[0] = C[0] / B[0]
    t[0] = (-1 * G[0]) / B[0]
    for i in range(1, n):
        s[i] = C[i] / (B[i] - (A[i] * s[i - 1]))
        t[i] = ((A[i] * t[i - 1]) - G[i]) / (B[i] - (A[i] * s[i - 1]))
    y[n - 1] = t[n - 1]
    for i in range(n - 2, -1, -1):
        y[i] = (s[i] * y[i + 1]) + t[i]
    return y


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


# def p(x, y):
#     return 3 * x + 2


# def f(x, y):
#     return -1 * x**2 * (2 + 6 * y) - y**2 * (12 * x + 4) * (1 + y)


# def mu(x, y):
#     return x**2 * y**2 * (1 + y)


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


# Коэффициенты первой системы
def A_1(x, y, i, j, hx):
    return (p(x[i] - (hx / 2), y[j])) / hx**2


def B_1(x, y, i, j, hx, tau):
    term1 = 2 / tau
    term2 = p(x[i] + (hx / 2), y[j]) / hx**2
    term3 = p(x[i] - (hx / 2), y[j]) / hx**2
    return term1 + term2 + term3


def C_1(x, y, i, j, hx):
    return (p(x[i] + (hx / 2), y[j])) / hx**2


def G_1(x, y, i, j, hy, tau, u):
    term1 = (2 * u[i][j]) / tau
    term2 = (q(x[i], y[j] + (hy / 2)) * (u[i][j + 1] - u[i][j])) / hy**2
    term3 = (q(x[i], y[j] - (hy / 2)) * (u[i][j] - u[i][j - 1])) / hy**2
    term4 = f(x[i], y[j])
    return (term1 + term2 - term3 + term4) * (-1)


# Коэффициенты второй системы
def A_2(x, y, i, j, hy):
    return (q(x[i], y[j] - (hy / 2))) / hy**2


def B_2(x, y, i, j, hy, tau):
    term1 = 2 / tau
    term2 = q(x[i], y[j] + (hy / 2)) / hy**2
    term3 = q(x[i], y[j] - (hy / 2)) / hy**2
    return term1 + term2 + term3


def C_2(x, y, i, j, hy):
    return (q(x[i], y[j] + (hy / 2))) / hy**2


def G_2(x, y, i, j, hx, tau, u):
    term1 = (2 * u[i][j]) / tau
    term2 = (p(x[i] + (hx / 2), y[j]) * (u[i + 1][j] - u[i][j])) / hx**2
    term3 = (p(x[i] - (hx / 2), y[j]) * (u[i][j] - u[i - 1][j])) / hx**2
    term4 = f(x[i], y[j])
    return (term1 + term2 - term3 + term4)*(-1)


def alternatingDirectionMethod(x, y, hx, hy, tau):
    k = 0
    exact = exactSolition(x, y)
    previous_U = fillBorderMatrix(x, y)

    U_0 = np.copy(previous_U)
    arrayOfU_K = [U_0]

    while (
        (normOfMatrix(arrayOfU_K[k] - exact) / normOfMatrix(arrayOfU_K[0] - exact))
        > EPS
    ):
        u_half = np.zeros((len(x), len(y)))
        current_U = np.zeros((len(x), len(y)))

        for i in range(len(x)):  # Условие на крайних столбцах
            u_half[i][0] = mu(x[i], 0)
            u_half[i][len(y) - 1] = mu(x[i], y[len(y) - 1])

        for j in range(1, len(y) - 1):
            term1 = np.zeros(len(x))
            term2 = np.zeros(len(x))
            term3 = np.zeros(len(x))
            term4 = np.zeros(len(x))

            term1[0] = 0
            term2[0] = -1
            term3[0] = 0
            term4[0] = mu(0, y[j])

            term1[len(x) - 1] = 0
            term2[len(x) - 1] = -1
            term3[len(x) - 1] = 0
            term4[len(x) - 1] = mu(x[len(x) - 1], y[j])
            for i in range(1, len(x) - 1):
                term1[i] = A_1(x, y, i, j, hx)
                term2[i] = B_1(x, y, i, j, hx, tau)
                term3[i] = C_1(x, y, i, j, hx)
                term4[i] = G_1(x, y, i, j, hy, tau, arrayOfU_K[k])
            linear_solve = tridiagonal_matrix_algorithm(term1, term2, term3, term4)
            for i in range(0, len(x)):
                u_half[i][j] = linear_solve[i]
            # Построили половинный шаг

        for j in range(len(y)):
            current_U[0][j] = mu(0, y[j])
            current_U[len(x) - 1][j] = mu(x[len(x) - 1], y[j])

        for i in range(1, len(x) - 1):
            lterm1 = np.zeros(len(y))
            lterm2 = np.zeros(len(y))
            lterm3 = np.zeros(len(y))
            lterm4 = np.zeros(len(y))

            lterm1[0] = 0
            lterm2[0] = -1
            lterm3[0] = 0
            lterm4[0] = mu(x[i], 0)

            lterm1[len(y) - 1] = 0
            lterm2[len(y) - 1] = -1
            lterm3[len(y) - 1] = 0
            lterm4[len(y) - 1] = mu(x[i], y[len(y) - 1])

            for j in range(1, len(y) - 1):
                lterm1[j] = A_2(x, y, i, j, hy)
                lterm2[j] = B_2(x, y, i, j, hy, tau)
                lterm3[j] = C_2(x, y, i, j, hy)
                lterm4[j] = G_2(x, y, i, j, hx, tau, u_half)
            linear_solve2 = tridiagonal_matrix_algorithm(lterm1, lterm2, lterm3, lterm4)

            for j in range(0, len(y)):
                current_U[i][j] = linear_solve2[j]
            # Построили k+1 решение
        arrayOfU_K.append(np.copy(current_U))
        k += 1
    return arrayOfU_K


x_0 = 0
x_n = 1

y_0 = 0
y_m = np.pi
# y_m = 1

N = 5
M = 15

step_x = (x_n) / N
step_y = (y_m) / M

x_i = np.arange(x_0, x_n + step_x, step_x)
y_i = np.arange(y_0, y_m + step_y, step_y)

c1 = 1
c2 = 3
# c1 = 2
# c2 = 5

d1 = 1
d2 = 1

sigma_1 = (
    c1 * (4 / (step_x**2)) * ((np.sin((np.pi * step_x) / 2 * x_i[len(x_i) - 1])) ** 2)
)
delta_1 = (
    c2 * (4 / (step_x**2)) * ((np.cos((np.pi * step_x) / 2 * x_i[len(x_i) - 1])) ** 2)
)

sigma_2 = (
    d1
    * (4 / (step_y**2))
    * ((np.sin((np.pi * step_y) / (2 * y_i[len(y_i) - 1]))) ** 2)
)
delta_2 = (
    d2
    * (4 / (step_y**2))
    * ((np.cos((np.pi * step_y) / (2 * y_i[len(y_i) - 1]))) ** 2)
)

sigma = min([sigma_1, sigma_2])
delta = max([delta_1, delta_2])

tauOpt = 2 / (np.sqrt(sigma * delta))

f_h = f_matrix(x_i, y_i)
exact_solve = exactSolition(x_i, y_i)
exact_solution_table = tabulate(exact_solve, tablefmt="fancy_grid")

array_triangle = alternatingDirectionMethod(x_i, y_i, step_x, step_y, tauOpt)

U_0 = fillBorderMatrix(x_i, y_i)
null_approx = normOfMatrix(calculate_lhu(U_0, x_i, y_i, step_x, step_y) + f_h)


data = []
headers = [
    "k",
    "||F-AU_k||",
    "rel.d",
    "||U^k - U_*||",
    "rel.error",
    "||U^k - U^(k-1)||",
    # "apost.est",
    "sp.red._k",
]

for k, solve in enumerate(array_triangle):
    data.append(
        [
            k,
            normOfMatrix(calculate_lhu(solve, x_i, y_i, step_x, step_y) + f_h),
            normOfMatrix(calculate_lhu(solve, x_i, y_i, step_x, step_y) + f_h)
            / null_approx,
            normOfMatrix(solve - exact_solve),
            normOfMatrix(solve - exact_solve) / normOfMatrix(U_0 - exact_solve),
            normOfMatrix(solve - array_triangle[k - 1]),
            # (spectr * normOfMatrix(solve - array_triangle[k - 1])) / (1 - spectr),
            normOfMatrix(solve - array_triangle[k - 1])
            / normOfMatrix(array_triangle[k - 1] - array_triangle[k - 2]),
        ]
    )

# Выведите таблицу
table1 = tabulate(data, headers, tablefmt="grid")

print(
    "\nПопеременно треугольный итерационный метод с чебышевскими параметрами. Вариант 8\n"
)
print(f"N = {N}; M = {M}\neps = {EPS}\n")
print(
    f"Мера аппроксимации ||F-AU_*|| = {normOfMatrix(calculate_lhu(exact_solve, x_i, y_i, step_x, step_y) + f_h)}"
)
print(f"Норма невязки нулевого приближения ||F-AU_0|| = {null_approx}")
# print(f"Число иттераций = {itterationApprox(eta)}")

# print(f"Спектральный радиус pho(H)= {spectr}")
print(table1)
# print(f"\nПриближенное решение:\n{approx_solition_table}")
# print(f"\nТочное решение:\n{exact_solution_table}")
