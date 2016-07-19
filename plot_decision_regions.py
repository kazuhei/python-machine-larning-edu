import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron


df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/iris/iris.data', header=None)
df.tail()

# 1-100行目の目的変数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1, Iris-virginicaを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)
# 1-100行目の1,3列目の抽出
X = df.iloc[0:100, [0, 2]].values
# # 品種 setosaのプロット
# plt.scatter(X[:50, 0], X[:50, 1],
#             color='red', marker='o', label='setosa')
# # 品種 versicolorのプロット
# plt.scatter(X[50:100, 0], X[50:100, 1],
#             color='blue', marker='x', label='versicolor')
# # 軸ラベルの設定
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# # 凡例の設定
# plt.legend(loc='upper left')
# # 図の表示
# plt.show()

# パーセプトロンのオブジェクト生成
ppn = Perceptron(eta=0.1, n_iter=10)
# トレーニングデータへのモデルの適合
ppn.fit(X, y)
# エポックと誤分類誤差の関係の折れ線グラフをプロット
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# # 軸のラベルの設定
# plt.xlabel('Epochs')
# plt.ylabel('Number of misclassifications')
# # 図の表示
# plt.show()

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # マーカーとカラーマップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット　
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
      np.arange(x2_min, x2_max, resolution))
    # 各特徴量を1次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線プロット
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # 軸の範囲設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
        marker=markers[idx], label=cl)

# 決定境界のプロット
plot_decision_regions(X, y, classifier=ppn)
# 軸のラベル設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の設定
plt.legend(loc='upper left')
# 図の表示
plt.show()

