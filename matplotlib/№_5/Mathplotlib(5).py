import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

labels = ['Apples', 'Bananas', 'Oranges', 'Grapes']
sizes = [25, 35, 20, 20]
colors = ['red', 'yellow', 'orange', 'purple']
explode = (0.1, 0, 0, 0)  # выделить сектор с яблоками

# Построение круговой диаграммы
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Доли различных фруктов')
plt.axis('equal')  # Чтобы диаграмма была круглой
plt.show()