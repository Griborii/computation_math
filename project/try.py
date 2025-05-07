import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_data(time_step):
    """
    Функция, генерирующая данные для кадра анимации.

    Аргументы:
        time_step: Номер кадра (время).

    Возвращает:
        Двумерный массив (матрицу) данных.
    """
    x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    z = np.sin(x**2+ time_step) / (x**2  + 1)
    return z

fig, ax = plt.subplots()
im = ax.imshow(generate_data(0), cmap='viridis', animated=True)  # Начальный кадр
fig.colorbar(im) # Добавляем цветовую шкалу

def update_fig(time_step):
    """
    Функция, обновляющая данные для каждого кадра анимации.

    Аргументы:
        time_step: Номер кадра (время).

    Возвращает:
        Кортеж, содержащий обновленное изображение.
    """
    data = generate_data(time_step)
    im.set_array(data)
    return im,

ani = animation.FuncAnimation(fig, update_fig, interval=50, blit=True)

plt.show() 