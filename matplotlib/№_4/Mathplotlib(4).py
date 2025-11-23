import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

data = {
    'Name': ['Анна', 'Антон', 'Мария', 'Макс', 'Макс', 'Алекс', 'Антон', 'Мария', 'Макс', 'Алекс', 'Антон', 'Анна'],
    'Points': [145, 2524, 343, 4212, 5212, 6421, 745, 842, 524, 3224, 212, 1214]
}

df = pd.DataFrame(data)

# Группировка по имени и вычисление статистик
grouped = df.groupby('Name')['Points'].agg(['min', 'max', 'mean'])

# Построение графика
plt.figure(figsize=(10, 6))
grouped.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Статистика очков игроков')
plt.xlabel('Игроки')
plt.ylabel('Очки')
plt.legend(['Минимальные', 'Максимальные', 'Средние'])
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Альтернативный вариант построения
fig, ax = plt.subplots(figsize=(10, 6))
grouped.plot(ax=ax, marker='o')
plt.title('Статистика очков игроков')
plt.xlabel('Игроки')
plt.ylabel('Очки')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()