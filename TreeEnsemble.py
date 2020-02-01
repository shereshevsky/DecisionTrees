import numpy as np

from DecisionTree import DecisionTree


class TreeEnsemble:
    def __init__(self, p_x, p_y, n_trees, sample_sz, min_leaf, max_depth=100):
        self.n_trees = n_trees
        self.sample_sz = sample_sz
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.trees = []
        self.oob_mses = []

        index = np.arange(0, p_y.size)
        for i in range(n_trees):
            selection = np.random.choice(index, size=sample_sz, replace=True)
            new_tree = DecisionTree(min_leaf, max_depth)
            new_tree.fit(p_x[selection], p_y[selection], selection)
            self.trees.append(new_tree)
            # Calculate OOB accuracy for the tree trained in current iteration
            oob_selection = np.delete(index, selection)
            # self.oob_mses.append(np.mean(np.square(new_tree.predict(p_x[oob_selection]) - p_y[oob_selection])))

        # The oob_mse function will compute the mean squared error over all out of bag (oob) samples.
        # That is, for each sample calculate the squared error using predictions from the trees that do not contain x
        # in their respective bootstrap sample, then average this score for all samples
        oob_results = []
        for item in index:
            predictions = [t.predict([p_x[item]]) for t in self.trees if item not in t.selection]
            if predictions:
                oob_results.append(np.square(np.mean(predictions) - p_y[item]))
        self.oob_mses.append(np.mean(oob_results))

    def predict(self, p_x):
        res = np.array([t.predict(p_x) for t in self.trees]).mean(axis=0)
        return res

    def oob_mse(self):
        return np.mean(self.oob_mses)
