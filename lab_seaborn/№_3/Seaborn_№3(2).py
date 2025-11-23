import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных titanic
titanic = sns.load_dataset('titanic')
plt.subplot(2, 2, 2)
sns.kdeplot(data=titanic, x='fare', hue='class', fill=True, common_norm=False,
            palette='Set2', alpha=0.6)
plt.title('Распределение стоимости билета по классам')
plt.xlim(0, 300)