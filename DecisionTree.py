import numpy as np
import json
from collections import defaultdict
from split_functions import variance_split


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class DecisionTree:
    def __init__(self, min_leaf, max_depth=100):
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.feature_importance = defaultdict(int)
        self.tree = None
        self.train_x = None
        self.train_y = None
        self.feature_names = None
        self.selection = None

    def __repr__(self):
        return json.dumps(self.tree, cls=NpEncoder)

    def learn_recursive(self, p_x, p_y, level=0):
        if (
                np.var(p_x, axis=0).sum() == 0
                or p_y.size <= self.min_leaf
                or np.var(p_y) == 0
                or level >= self.max_depth
        ):
            return {"label": np.round(np.mean(p_y), 5), "cnt": p_y.size}

        # generate list of best variance reduction for each feature and find the best feature and position to split by
        (split_value, best_var_reduction), best_feature_index = sorted(
            [(variance_split(p_x[:, i], p_y, self.min_leaf), i) for i in range(p_x.shape[1])],
            key=lambda _x: _x[0][1],
            reverse=True)[0]

        # no good variance reduction options found -> return average
        if not best_var_reduction:
            return {"label": np.round(np.mean(p_y), 5), "cnt": p_y.size}

        l_mask = p_x[:, best_feature_index] < split_value

        curr_feature = self.feature_names[best_feature_index]
        self.feature_importance[curr_feature] += best_var_reduction

        l_node = self.learn_recursive(p_x[l_mask], p_y[l_mask], level + 1)
        be_node = self.learn_recursive(p_x[~l_mask], p_y[~l_mask], level + 1)

        return {
            "split_feature": curr_feature,
            "split_value": split_value,
            "<": l_node,
            ">=": be_node,
        }

    def fit(self, p_x, p_y, selection):
        self.selection = selection
        self.train_x = p_x
        self.train_y = p_y
        self.feature_names = list(range(p_x.shape[1]))
        self.tree = self.learn_recursive(p_x, p_y)

    def mse(self):
        predicted = self.predict(self.train_x)
        return np.mean(np.square(np.array(predicted) - self.train_y))

    def predict(self, p_x):
        res = []
        for i in p_x:
            path = self.tree
            if not path:
                raise Exception("Fit the model first")
            while path and "label" not in path:
                path = (
                    path["<"]
                    if i[path["split_feature"]] < path["split_value"]
                    else path[">="]
                )
            res.append(path["label"])
        return res
