import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных titanic
titanic = sns.load_dataset('titanic')
plt.subplot(2, 2, 3)
# Для lmplot создаем отдельный график
sns.lmplot(data=titanic, x='age', y='fare', hue='sex', col='class',
           height=4, aspect=1.2, scatter_kws={'alpha':0.6})
plt.suptitle('Зависимость возраста и стоимости билета', y=1.02)
plt.show()