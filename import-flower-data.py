import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/iris/iris.data', header=None)
df.tail()

# 1-100行目の目的変数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1, Iris-virginicaを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)
# 1-100行目の1,3列目の抽出
X = df.iloc[0:100, [0, 2]].values
# 品種 setosaのプロット
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
# 品種 versicolorのプロット
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')
# 軸ラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の設定
plt.legend(loc='upper left')
# 図の表示
plt.show()
