import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных iris
iris = sns.load_dataset('iris')

# Диаграмма парных зависимостей
sns.pairplot(iris, hue='species', palette='viridis')
plt.suptitle('Pairplot для набора данных Iris', y=1.02)
plt.show()