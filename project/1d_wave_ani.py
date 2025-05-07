from funcs import *

t_start = 0
t_end = 1
x_start = 0
x_end = 1

start_func = gaus
g = 4
H = 1
c = math.sqrt(g * H)

dx = 0.01
dt = 0.01
res = wave_1d(g, H, start_func, dt, dx, t_start, t_end, x_start, x_end)
vals, x_grid, t_grid = res["result"]["values"], res["result"]["x_grid"], res["result"]["t_grid"]
h_vals = []
for i in range(len(vals)):
    h_vals.append(np.copy(vals[i][1]))

print(f"kurant number is: {c * dt / dx}")
wave_1d_animation(h_vals, x_grid)


# тут можно посмотреть аналитическое решение (разницы не заметно)
# theor_vals = wave_eq_solve(c, func_sim(start_func, x_start, x_end), x_grid, t_grid)
# wave_animation(theor_vals, x_grid)
