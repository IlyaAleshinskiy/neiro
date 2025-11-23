import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

x = np.linspace(-3, 3, 51)

# Создание фигуры с двумя подграфиками
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Первый график - экспоненциальная функция
ax[0].plot(x, np.exp(x), 'r-', linewidth=2)
ax[0].set_title('Экспоненциальная функция y = e^x')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].grid(True)

# Второй график - натуральный логарифм (только для положительных x)
x_positive = x[x > 0]  # Логарифм определен только для положительных значений
ax[1].plot(x_positive, np.log(x_positive), 'b-', linewidth=2)
ax[1].set_title('Натуральный логарифм y = ln(x)')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].grid(True)

plt.tight_layout()
plt.show()