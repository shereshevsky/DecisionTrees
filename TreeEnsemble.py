import numpy as np

from DecisionTree import DecisionTree


class TreeEnsemble:
    def __init__(self, p_x, p_y, n_trees, sample_sz, min_leaf, max_depth=5):
        self.n_trees = n_trees
        self.sample_sz = sample_sz
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.trees = []

        index = np.arange(0, p_y.size)
        for i in range(n_trees):
            selection = np.random.choice(index, size=sample_sz, replace=True)
            new_tree = DecisionTree(min_leaf, max_depth)
            new_tree.fit(p_x[selection], p_y[selection])
            self.trees.append(new_tree)

    def predict(self, p_x):
        res = np.array([t.predict(p_x) for t in self.trees]).mean(axis=0)
        return res

    def oob_mse(self):
        pass
