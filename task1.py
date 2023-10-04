import numpy as np
from tabulate import tabulate

EPS = 0.001

np.set_printoptions(formatter={"all": lambda x: f"{x:.4f}"})  # Set the format


def spectralRadiusApprox(xi):
    return (1 - xi) / (1 + xi)


def itterationApprox(ksi):
    return np.log(1 / EPS) / (2 * ksi)


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


def calculate_lhu(u, x, y, hx, hy):
    lhu = np.zeros((len(x), len(y)))
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            term1 = p(x[i] + hx / 2, y[j]) * (u[i + 1][j] - u[i][j]) / hx**2
            term2 = p(x[i] - hx / 2, y[j]) * (u[i][j] - u[i - 1][j]) / hx**2
            term3 = q(x[i], y[j] + hy / 2) * (u[i][j + 1] - u[i][j]) / hy**2
            term4 = q(x[i], y[j] - hy / 2) * (u[i][j] - u[i][j - 1]) / hy**2
            lhu[i][j] = term1 - term2 + term3 - term4
    return lhu


def simpleItteration(x, y, step_x, step_y, iter):
    k = 0
    previous_U = fillBorderMatrix(x, y)
    current_U = fillBorderMatrix(x, y)
    U_0 = np.copy(previous_U)
    arrayOfU_K = [U_0]

    while k < iter:
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
        arrayOfU_K.append(np.copy(current_U))
    return arrayOfU_K, k


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

f_matrix = f_matrix(x_i, y_i)
exact_solve = exactSolition(x_i, y_i)
U_0 = fillBorderMatrix(x_i, y_i)
print("\nМетод простых иттераций. Вариант 8\n")
print(f"N = {N}; M = {M}\neps = {EPS}\n")
print(
    f"Мера аппроксимации ||F-AU_*|| = {normOfMatrix(calculate_lhu(exact_solve, x_i, y_i, step_x, step_y) +f_matrix)}"
)
null_approx = normOfMatrix(calculate_lhu(U_0, x_i, y_i, step_x, step_y) + f_matrix)
print(f"Норма невязки нулевого приближения ||F-AU_0|| = {null_approx}")
print(f"Число иттераций = {itterationApprox(ksi)}")
spectr = spectralRadiusApprox(ksi)
print(f"Спектральный радиус pho(H)= {spectr}")

array_simple_itteration_matrix, amountItter = simpleItteration(
    x_i, y_i, step_x, step_y, itterationApprox(ksi)
)
approx_solition_table = tabulate(
    array_simple_itteration_matrix[-1], tablefmt="fancy_grid"
)


exact_solution_table = tabulate(exact_solve, tablefmt="fancy_grid")


data = []
headers = [
    "k",
    "||F-AU_k||",
    "rel.d",
    "||U^k - U_*||",
    "rel.error",
    "||U^k - U^(k-1)||",
    "apost.est",
    "sp.red._k",
]

for k, solve in enumerate(array_simple_itteration_matrix):
    data.append(
        [
            k,
            normOfMatrix(calculate_lhu(solve, x_i, y_i, step_x, step_y) + f_matrix),
            normOfMatrix(calculate_lhu(solve, x_i, y_i, step_x, step_y) + f_matrix)
            / null_approx,
            normOfMatrix(solve - exact_solve),
            normOfMatrix(solve - exact_solve) / normOfMatrix(U_0 - exact_solve),
            normOfMatrix(solve - array_simple_itteration_matrix[k - 1]),
            (spectr * normOfMatrix(solve - array_simple_itteration_matrix[k - 1]))
            / (1 - spectr),
            normOfMatrix(solve - array_simple_itteration_matrix[k-1])/normOfMatrix(array_simple_itteration_matrix[k-1]-array_simple_itteration_matrix[k-2]),
        ]
    )

# Выведите таблицу
table1 = tabulate(data, headers, tablefmt="grid")


print(table1)
print(f"\nПриближенное решение:\n{approx_solition_table}")
print(f"\nТочное решение:\n{exact_solution_table}")
