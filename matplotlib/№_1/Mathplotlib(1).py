import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

# Линейный график зависимости квадрата числа от самого числа
x = np.linspace(-10, 10, 100)
y = x**2

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Зависимость квадрата числа от самого числа')
plt.xlabel('x')
plt.ylabel('x²')
plt.grid(True)
plt.show()