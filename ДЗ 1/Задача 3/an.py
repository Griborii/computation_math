from scipy.optimize import fsolve
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Вспомогательная функция
def scal_prod(x, y):
    if len(x) != len(y):
        raise
    return sum([x[i]*y[i] for i in range(len(x))])


def implicit_runge_kutt(a, b, c, n, start, finish, start_val, func):
    h = (finish - start) / n
    val = np.copy(start_val)
    t = start
    vals = [val]
    time_grid = [0]
    for i in range(n):
        val_mid = [val] # промежуточные значения функции
        t_mid = [t] # промежуточные значения времени
        for j in range(1, len(b)):
            t_mid.append(t + c[j] * h)
            const1 = val + h * sum([a[j][k] * func(val_mid[k], t_mid[k]) for k in range(j)])
            # print("const:", const1)
            def equation(x):
                return x - func(x, t + c[j] * h) * a[j][j] * h - const1
            slve = fsolve(equation, x0=val)
            # print(f"solve: {slve}, solve_prec: {slve - func(slve, t + c[j] * h) * a[j][j] - const1}")
            val_mid.append(np.copy(slve))
        t += h
        time_grid.append(t)
        dif = [func(val_mid[k], t_mid[k]) for k in range(len(b))]
        # print("bubu: ", h, scal_prod(b, dif))
        val += h * scal_prod(b, dif)
        vals.append(np.copy(val))
        # print(f"time: {t}")
        # print(f"dif: {dif}")
        # print("val: ", val)
    return (vals, time_grid)


def euler_method(n, start, finish, srart_val, func):
    h = (finish - start) / n
    val = np.copy(srart_val)
    t = start
    for i in range(n):
        val += h * func(val, t)
        print(f"time: {t}")
        print("val: ", val)
        t += h
    return val


def k_1(t):
    return 0.01 * max(0, math.sin(2*math.pi*t/(24*60*60)))

k = [k_1, 10**5, 10**(-16)]

def func2(c, t):
    a_1 = k[0](t)*c[2]
    a_2 = k[1] * c[0]
    a_3 = k[2]*c[1]*c[3]
    # print(a_1, a_2, a_3)
    return np.copy(np.array([a_1 - a_2, a_1 - a_3, a_3 - a_1, a_2 - a_3]))

a = [
     [1, 0, 0],
     [0, 1/3, 1],
     [-1/12, 3/4, 1/3]
]
b = [-1/12, 3/4, 1/3]
c = [1, 1/3, 1]

T = 24 * 60 * 60.0
start_val = np.array([0.0, 0.0, 5e11, 8e11], dtype=np.float64)
steps = 100

def fnc(x, t):
    return -x

# implicit_runge_kutt(a, b, c, 10, 0, 1, np.array([1.0]), fnc)
vals, time_grid = implicit_runge_kutt(a, b, c, steps, 0, T, start_val, func2)
print(len(time_grid), len(vals))
res = []
for i in range(4):
    res.append([])
    for j in range(len(vals)):
        res[i].append(vals[j][i])

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))  # figsize=(ширина, высота)
plt.tight_layout(pad=8.0)

for i in range(2):
    for j in range(2):
        ind = i * 2 + j
        axs[i, j].plot(res[ind], time_grid)
        axs[i, j].set_title(f"c_{ind}(t)")
        axs[i, j].set_xlabel('t')
        axs[i, j].set_ylabel(f"c_{ind}")

plt.legend()
plt.show()
# animation_html = explicit_runge_kutt_animated(a, b, c, 1000, T / 1000, 0, start_val, func2)
# animation_html
# print(euler_method(steps, 0, T * 2, start_val, func2))