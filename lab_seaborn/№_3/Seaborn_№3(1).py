import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных titanic
titanic = sns.load_dataset('titanic')
plt.figure(figsize=(12, 8))

# Подготовка данных для корреляции (только числовые колонки)
numeric_cols = titanic.select_dtypes(include=['number']).columns
corr_matrix = titanic[numeric_cols].corr()

plt.subplot(2, 2, 1)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', linewidths=0.5)
plt.title('Тепловая карта корреляций (Titanic)')
