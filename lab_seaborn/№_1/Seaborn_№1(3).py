import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных
tips = sns.load_dataset('tips')

plt.subplot(2, 2, 3)
sns.boxplot(data=tips, x='day', y='total_bill', palette='Set2')
plt.title('Распределение total_bill по дням недели')
plt.tight_layout()
plt.show()