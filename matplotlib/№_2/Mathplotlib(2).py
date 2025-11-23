import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

t = np.linspace(0, 10, 51)

# Построение синусоиды и косинусоиды
plt.figure(figsize=(10, 6))

# Синусоида
plt.plot(t, np.sin(t), 'b-', linewidth=2, label='sin(t)')
# Косинусоида
plt.plot(t, np.cos(t), 'r--', linewidth=2, label='cos(t)')

plt.title('Синусоида и косинусоида')
plt.xlabel('t')
plt.ylabel('Значения')
plt.legend()
plt.grid(True)
plt.show()

# Создание массива f
f = np.cos(t) + np.sin(t)

# Построение линейной диаграммы
plt.figure(figsize=(10, 6))
plt.plot(t, f, 'g-', linewidth=2)
plt.title('График f(t)')
plt.xlabel('Значения t')
plt.ylabel('Значения f')
plt.xlim(0.5, 9.5)
plt.ylim(-2.5, 2.5)
plt.grid(True)
plt.show()