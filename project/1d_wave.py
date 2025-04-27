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
    # print("pace is: ", h)
    val = np.copy(start_val)
    vals = [np.copy(val)]
    t = start
    for i in range(n):
        val_mid = [val]
        t_mid = [t]
        for j in range(1, len(b)):
            t_mid.append(t + c[j] * h)
            val_mid.append(val + h * sum([a[j][k] * func(val_mid[k], t_mid) for k in range(j)]))
        t += h
        dif = [func(val_mid[k], t_mid[k]) for k in range(len(b))]
        vals.append(np.copy(val))
        val += h * sum(np.array([b[i]*dif[i] for i in range(len(dif))]))
    return vals

def wave(g, H, func, dt, dx, t_start, t_end, x_start, x_end):
    # калибровка длины шага, чтобы из вмещалось целое число
    N_t = int((t_end - t_start) / dt)
    dt = (t_end - t_start) / N_t
    N_x = int((x_end - x_start) / dx)
    dx = (x_end - x_start) / N_x
    c = math.sqrt(g * H)

    # начальный вектор значений функции при x от 0 до 1 при t = 0
    x = [x_start + i * dx for i in range(N_x + 1)]
    t = [t_start + i * dx for i in range(N_t + 1)]
    f_0 = [[0 for i in range(N_x + 1)], [func(x[i]) for i in range(N_x + 1)]]

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

def wave_animation(rk_val, x_grid):
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
    
def wave_eq_solve(c, start_func, x_grid, t_grid):
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

dt = 0.001
dx = 0.001
t_start = 0
t_end = 1
x_start = 0
x_end = 1

# калибровка длины шага, чтобы из вмещалось целое число
N_t = int((t_end - t_start) / dt)
dt = (t_end - t_start) / N_t
N_x = int((x_end - x_start) / dx)
dx = (x_end - x_start) / N_x

start_func = gaus
g = 5
H = 1
c = math.sqrt(g * H)

res = wave(g, H, start_func, dt, dx, t_start, t_end, x_start, x_end)
vals, x_grid, t_grid = res["result"]["values"], res["result"]["x_grid"], res["result"]["t_grid"]
h_vals = []
for i in range(len(vals)):
    h_vals.append(np.copy(vals[i][1]))

# wave_animation(vals, x_grid)
theor_vals = wave_eq_solve(c, func_sim(start_func, x_start, x_end), x_grid, t_grid)
err = [np.max(h_vals[i] - theor_vals[i]) for i in range(len(t_grid))]
# wave_animation(h_vals, x_grid)
plt.plot(t_grid, err)
plt.show()