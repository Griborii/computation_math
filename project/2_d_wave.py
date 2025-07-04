import matplotlib.animation as animation
import time
from IPython.display import HTML
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
import warnings

def explicit_runge_kutt(n, h, start, start_val, func):
    a = [
     [0, 0, 0, 0],
     [0.5, 0, 0, 0],
     [0, 0.5, 0, 0],
     [0, 0, 1, 0]
    ]
    b = [1/6, 1/3, 1/3, 1/6]
    c = [0, 0.5, 0.5, 1]
    val = np.copy(start_val)
    vals = [np.copy(val)]
    t = start
    for i in range(n):
        val_mid = [val] # промежуточные значения функции
        t_mid = [t] # промежуточные значения времени
        for j in range(1, len(b)):
            t_mid.append(t + c[j] * h)
            val_mid.append(val + h * sum([a[j][k] * func(val_mid[k], t_mid[k]) for k in range(j)]))
        t += h
        dif = [func(val_mid[k], t_mid[k]) for k in range(len(b))]
        val += h * sum(np.array([b[i] * dif[i] for i in range(len(b))]))
        vals.append(np.copy(val))
    return vals

def wave_2d(g, H, f, func, dt, dx, dy, t_start, t_end, x_start, x_end, y_start, y_end):
    # калибровка длины шага, чтобы из вмещалось целое число
    st_param = dt * g * H * (1 / dx**2 + 1 / dy**2)
    print(f"st_param = {st_param}")
    if st_param > 2.8:
        warnings.warn(f"Это может быть не устойчиво, st_param = {st_param} > 2.8", UserWarning)
    N_t = int((t_end - t_start) / dt)
    dt = (t_end - t_start) / N_t
    N_x = int((x_end - x_start) / dx)
    dx = (x_end - x_start) / N_x
    N_y = int((y_end - y_start) / dy)
    dy = (y_end - y_start) / N_y
    c = math.sqrt(g * H)

    # начальный вектор значений функции при x от 0 до 1 при t = 0
    x_y = [[(x_start + j * dx, y_start + i * dy) for i in range(N_y)]
           for j in range(N_y)]
    t = [t_start + i * dt for i in range(N_t + 1)]
    f_0 = [
        np.zeros((N_x, N_y)),
        np.zeros((N_x, N_y)),
        np.zeros((N_x, N_y))
        ]
    for i in range(N_x):
        for j in range(N_y):
            f_0[2][i][j] = func(i * dx, j * dy)
    # вместо правой части производная по координате
    def minus_x_y_diff(matr, t):
        diff = [
            np.zeros((N_x, N_y)),
            np.zeros((N_x, N_y)),
            np.zeros((N_x, N_y))
            ]
        for i in range(N_x):
            for j in range(N_y):
                i_min_1 = (i - 1 + N_x) % N_x
                i_pl_1 = (i + 1 + N_x) % N_x
                j_min_1 = (j - 1 + N_y) % N_y
                j_pl_1 = (j + 1 + N_y) % N_y
                diff[0][i][j] = -g * (matr[2][i_pl_1][j] - matr[2][i_min_1][j]) / (2 * dx)
                diff[0][i][j] += f * matr[1][i][j]
                diff[1][i][j] = -g * (matr[2][i][j_pl_1] - matr[2][i][j_min_1]) / (2 * dy)
                diff[1][i][j] -= f * matr[0][i][j]
                diff[2][i][j] = -H * (matr[0][i_pl_1][j] - matr[0][i_min_1][j]) / (2 * dx)
                diff[2][i][j] += -H * (matr[1][i][j_pl_1] - matr[1][i][j_min_1]) / (2 * dy)
        return np.array([diff[0], diff[1], diff[2]])

    # МРК 4 для шага по времени
    # вместо правой части в explicit_runge_kutt кидаем производную по координате
    rk_res =  explicit_runge_kutt(N_t, dt, t_start, f_0, minus_x_y_diff) 
    return {"result": {"values": rk_res, "x_y_grid": x_y, "t_grid": t}}

def wave_2d_animation(rk_val, x_y_grid):

    def generate_data(time_step):
       return rk_val[time_step]

    fig, ax = plt.subplots()
    im = ax.imshow(generate_data(0), cmap='viridis', animated=True)  # Начальный кадр
    fig.colorbar(im) # Добавляем цветовую шкалу

    def update_fig(time_step):
        data = generate_data(time_step)
        im.set_array(data)
        return im,

    ani = animation.FuncAnimation(fig, update_fig, interval=100, blit=True)
    plt.show()  
# def wave_1d_eq_solve(c, start_func, x_grid, t_grid):
#     theor_vals = []
#     for i in range(len(t_grid)):
#         theor_vals.append(np.zeros(len(x_grid)))
#         t = t_grid[i]
#         for j in range(len(x_grid)):
#             x = x_grid[j]
#             theor_vals[i][j] = (start_func(x + c * t) + start_func(x - c * t)) / 2
#     return np.array(theor_vals)

# def func_sim(func, x_left, x_right):
#     def new_func(x):
#         T = x_right - x_left
#         x_sim = x - math.floor((x - x_left) / T) * T
#         return gaus(x_sim)
    
#     return new_func

def x_gaus(x, y):
    d = 0.1
    return math.exp(-((x - 0.5) / d) ** 2)

def rad_gaus(x, y):
    d = 0.1
    return math.exp(-((x - 0.5)**2 + (y - 0.5)**2) / d**2)

t_start = 0
t_end = 1
x_start = 0
x_end = 1
y_start = 0
y_end = 1

start_func = rad_gaus
f = 40
g = 1
H = 1
c = math.sqrt(g * H)

dx = 0.02
dy = 0.02
dt = 0.0025
res = wave_2d(g, H, f, start_func, dt, dx, dy, t_start, t_end, x_start, x_end, y_start, y_end)
vals, x_y_grid, t_grid = res["result"]["values"], res["result"]["x_y_grid"], res["result"]["t_grid"]
h_vals = []
for i in range(len(vals)):
    h_vals.append(np.copy(vals[i][2]))

wave_2d_animation(h_vals, x_y_grid)
# print(f"kurant number is: {c * dt / dx}")
# wave_animation(h_vals, x_grid)

# тут можно посмотреть аналитическое решение (разницы не заметно)
# theor_vals = wave_eq_solve(c, func_sim(start_func, x_start, x_end), x_grid, t_grid)
# wave_animation(theor_vals, x_grid)
