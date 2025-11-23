import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных
tips = sns.load_dataset('tips')

g = sns.FacetGrid(tips, col='time', row='sex', height=4)
g.map(sns.regplot, 'total_bill', 'tip')
g.add_legend()
plt.show()