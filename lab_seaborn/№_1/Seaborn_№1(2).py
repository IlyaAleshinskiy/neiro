import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных
tips = sns.load_dataset('tips')

plt.subplot(2, 2, 2)
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex')
plt.title('Зависимость total_bill и tip по полу')