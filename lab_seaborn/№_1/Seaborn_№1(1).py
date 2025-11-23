import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных
tips = sns.load_dataset('tips')

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
sns.histplot(data=tips, x='total_bill', kde=True)
plt.title('Распределение total_bill с KDE')