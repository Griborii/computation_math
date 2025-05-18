import matplotlib.animation as animation
import time
from IPython.display import HTML
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

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

def wave_1d(g, H, func, dt, dx, t_start, t_end, x_start, x_end):
    # калибровка длины шага, чтобы из вмещалось целое число
    N_t = int((t_end - t_start) / dt)
    dt = (t_end - t_start) / N_t
    N_x = int((x_end - x_start) / dx)
    dx = (x_end - x_start) / N_x
    c = math.sqrt(g * H)

    # начальный вектор значений функции при x от 0 до 1 при t = 0
    x = [x_start + i * dx for i in range(N_x + 1)]
    t = [t_start + i * dt for i in range(N_t + 1)]
    f_0 = [np.array([0 for i in range(N_x + 1)]), np.array([func(x[i]) for i in range(N_x + 1)])]
    # вместо правой части производная по координате
    def minus_x_diff(vec, t):
        diff = [np.zeros(len(vec[0])), np.zeros(len(vec[0]))]
        diff[0][0] = (vec[1][1] - vec[1][-2]) / (2 * dx)
        diff[0][-1] = diff[0][0]
        for i in range(1, len(vec[1]) - 1):
            diff[0][i] = (vec[1][i + 1] - vec[1][i - 1]) / (2 * dx)
        diff[1][0] = (vec[0][1] - vec[0][-2]) / (2 * dx)
        diff[1][-1] = diff[1][0]
        for i in range(1, len(vec[0]) - 1):
            diff[1][i] = (vec[0][i + 1] - vec[0][i - 1]) / (2 * dx)
        return np.array([-diff[0] * g, -diff[1] * H])

    # МРК 4 для шага по времени
    # вместо правой части в explicit_runge_kutt кидаем производную по координате
    rk_res =  explicit_runge_kutt(N_t, dt, t_start, f_0, minus_x_diff) 
    return {"result": {"values": rk_res, "x_grid": x, "t_grid": t}}

def wave_1d_animation(rk_val, x_grid):
    # анимация
    fig, ax = plt.subplots()
    line, = ax.plot(x_grid, rk_val[0])  # Инициализируем линию графика
    ax.set_xlim(0, 1)  # Задаём пределы по x
    ax.set_ylim(0, 1)  # Задаём пределы по y (важно для стабильности)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x, t)")
    ax.set_title("Анимация функции f(x, t)")

    def animate(ind):
        """Функция, вызываемая на каждом кадре анимации."""
        y = rk_val[int(ind)] # Вычисляем новое значение функции
        line.set_ydata(y)      # Обновляем данные линии графика
        return line,           # Важно вернуть итерируемый объект (tuple)

    ani = animation.FuncAnimation(
        fig, animate, frames=np.linspace(0, len(rk_val) - 1, len(rk_val) - 1),  # 100 кадров от 0 до 20
        interval=50, blit=False, repeat=True
    )
    plt.show()
    
def wave_1d_eq_solve(c, start_func, x_grid, t_grid):
    theor_vals = []
    for i in range(len(t_grid)):
        theor_vals.append(np.zeros(len(x_grid)))
        t = t_grid[i]
        for j in range(len(x_grid)):
            x = x_grid[j]
            theor_vals[i][j] = (start_func(x + c * t) + start_func(x - c * t)) / 2
    return np.array(theor_vals)

def func_sim(func, x_left, x_right):
    def new_func(x):
        T = x_right - x_left
        x_sim = x - math.floor((x - x_left) / T) * T
        return gaus(x_sim)
    
    return new_func

def gaus(x):
    d = 0.1
    return math.exp(-((x - 0.5) / d) ** 2)

def wave_1d_SBP(g, H_0, func, dt, dx, t_start, t_end, x_start, x_end):
    # калибровка длины шага, чтобы из вмещалось целое число
    N_t = int((t_end - t_start) / dt)
    dt = (t_end - t_start) / N_t
    N_x = int((x_end - x_start) / dx)
    dx = (x_end - x_start) / N_x
    c = math.sqrt(g * H_0)

    # начальный вектор значений функции при x от 0 до 1 при t = 0
    x = [x_start + i * dx for i in range(N_x + 1)]
    t = [t_start + i * dt for i in range(N_t + 1)]
    f_0 = [np.array([0 for i in range(N_x + 1)]), np.array([func(x[i]) for i in range(N_x + 1)])]

    H = np.eye(N_x + 1)
    H[0][0], H[1][1], H[2][2], H[3][3] = (17/48, 59/48, 43/48, 49/48)
    for i in range(4):
        H[N_x - i][N_x - i] = H[i][i]
    Q = np.zeros((N_x + 1, N_x + 1))
    Q[0][0], Q[0][1], Q[0][2], Q[0][3], Q[1][2], Q[1][3], Q[2][3], Q[2][4], Q[3][4], Q[3][5] = (
        -1/2, 59/96, -1/12, -1/32, 59/96, 0.0, 59/96, -1/12, 2/3, -1/12
    )
    for i in range(4):
        for j in range(0, i):
            Q[i][j] = - Q[j][i]
    
    for i in range(4, N_x - 3):
        Q[i][i-2:i+3] = (1/12, -2/3, 0.0, 2/3, -1/12)
    
    for i in range(4):
        for j in range(6):
            Q[N_x - i][N_x - j] = - Q[i][j]
    
    D = np.linalg.inv(H) @ Q
    # вместо правой части производная по координате
    def minus_x_diff(vec, t):
        u = np.copy(vec[0])
        h = np.copy(vec[1])
        e_1 = np.zeros(N_x + 1)
        e_1[0] = 1
        e_n = np.zeros(N_x + 1)
        e_n[N_x] = 1
        return np.array([-g*D@h / dx, H_0*(-D@u - u[0]*e_1 + u[N_x]*e_n) / dx])

    # МРК 4 для шага по времени
    # вместо правой части в explicit_runge_kutt кидаем производную по координате
    rk_res =  explicit_runge_kutt(N_t, dt, t_start, f_0, minus_x_diff) 
    return {"result": {"values": rk_res, "x_grid": x, "t_grid": t}}
