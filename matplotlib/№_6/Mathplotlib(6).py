import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

stores = ['Store A', 'Store B', 'Store C', 'Store D', 'Store E']
sales = [50, 80, 60, 70, 90]

# Построение вертикальной столбчатой диаграммы
plt.figure(figsize=(10, 6))
bars = plt.bar(stores, sales, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet'])

# Добавление значений на столбцы
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}', ha='center', va='bottom')

plt.title('Количество проданных товаров по магазинам')
plt.xlabel('Магазины')
plt.ylabel('Количество продаж')
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()