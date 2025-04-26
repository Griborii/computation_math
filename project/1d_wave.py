import matplotlib.animation as animation
import time
from IPython.display import HTML
import math
import numpy as np
import matplotlib.pyplot as plt

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

def gaus(x):
    d = 0.1
    return math.exp(-((x - 0.5) / d) ** 2)

def wave(func, dt, dx):
    # калибровка длины шага, чтобы из вмещалось целое число
    t_start = 0
    t_end = 1
    x_start = 0
    x_end = 1
    N_t = int((t_end - t_start) / dt)
    dt = (t_end - t_start) / N_t
    N_x = int((x_end - x_start) / dx)
    dx = (x_end - x_start) / N_x

    # начальный вектор значений функции при x от 0 до 1 при t = 0
    x = [x_start + i * dx for i in range(N_t + 1)]
    f_0 = [[0 for i in range(N_t + 1)], [func(x[i]) for i in range(N_t + 1)]]
    
    # анимация
    fig, ax = plt.subplots()
    line, = ax.plot(x, f_0[1])  # Инициализируем линию графика
    ax.set_xlim(0, 1)  # Задаём пределы по x
    ax.set_ylim(0, 1)  # Задаём пределы по y (важно для стабильности)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x, t)")
    ax.set_title("Анимация функции f(x, t)")

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
        return np.array([-diff[0], -diff[1]])

    def animate(ind):
        """Функция, вызываемая на каждом кадре анимации."""
        y = rk_res[int(ind)][1] # Вычисляем новое значение функции
        line.set_ydata(y)      # Обновляем данные линии графика
        return line,           # Важно вернуть итерируемый объект (tuple)

    # МРК 4 для шага по времени
    # вместо правой части в explicit_runge_kutt кидаем производную по координате
    rk_res =  explicit_runge_kutt(N_t, dt, t_start, f_0, minus_x_diff)

    ani = animation.FuncAnimation(
        fig, animate, frames=np.linspace(0, len(rk_res) - 1, len(rk_res) - 1),  # 100 кадров от 0 до 20
        interval=50, blit=False, repeat=True
    )

    plt.show()

wave(gaus, 0.01, 0.01)
