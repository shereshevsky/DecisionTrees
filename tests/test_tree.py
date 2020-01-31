import numpy as np

from DecisionTree import DecisionTree

# X = np.array([[0, 5, 33],
#               [3, 5, 4],
#               [2, 2, 2]
#               ])
#
# y = np.array([1, 0, 0])
#
# dt = DecisionTree(2)
# dt.fit(X, y)
#
# res = dt.predict(np.array([[0, 5, 1], [2, 3, 2], [1, 2, 3], [1, 5, 4], [1, 6, 8]]))
#
# print(dt)
# print(res)
#
# X = np.array([[1, 6, 2],
#               [1, 6, 2],
#               [1, 5, 1],
#               [1, 5, 1]])
# y = np.array([1, 1, 1, 1])
#
# dt = DecisionTree(2)
# dt.fit(X, y)
#
# res = dt.predict(np.array([[0, 5, 1], [2, 3, 2], [1, 2, 3], [1, 5, 4], [1, 6, 8]]))
# print(res)
#
# X = np.array([[4, 5, 3],
#               [1, 5, 1],
#               [3, 5, 4],
#               [3, 5, 4]])
# y = np.array([1, 1, 0, 0])
# dt = DecisionTree(2)
# dt.fit(X, y)
#
# res = dt.predict(np.array([[0, 5, 1], [2, 3, 2], [1, 2, 3], [1, 5, 4], [1, 6, 8]]))
# print(dt)
# print(res)
#
# X = np.array([[1, 6, 2],
#        [1, 5, 1],
#        [1, 6, 2],
#        [1, 5, 1]])
# y = np.array([1, 1, 1, 1])
#
# dt = DecisionTree(2)
# dt.fit(X, y)
#
# res = dt.predict(np.array([[0, 5, 1], [2, 3, 2], [1, 2, 3], [1, 5, 4], [1, 6, 8]]))
# print(dt)
# print(res)
#
# X = np.array([[4, 5, 3],
#        [4, 5, 3],
#        [4, 5, 3],
#        [3, 5, 4]])
# y = np.array([1, 1, 1, 0])
#
# dt = DecisionTree(2)
# dt.fit(X, y)
#
# res = dt.predict(np.array([[4, 5, 1], [2, 3, 2], [1, 2, 3], [1, 5, 4], [99, 6, 8]]))
# print(dt)
# print(res)
#
# print(dt.mse())

### cricket

y = np.array([  1,    0,      0,      1,      0,      1,      1,      0,      1])
X = np.array([
                [1,9],[1,8], [0,8],   [1,8],  [0,9],  [1,9],  [1,8],  [0,9],  [1,8]])

dt = DecisionTree(2, 1)
dt.fit(X, y)
print(dt)

###  Boston
from sklearn.datasets import load_boston
from sklearn.preprocessing.data import normalize
boston = load_boston()
boston_y = boston.target
boston_X = boston.data

dt = DecisionTree(5, 10)
dt.fit(normalize(boston_X,  axis=0), boston_y)
print(dt)
print(dt.feature_importance)
print(dt.mse())

