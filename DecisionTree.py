import json
import numpy as np

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
    def __init__(self, min_leaf, max_depth=3):
        self.tree = None
        self.min_leaf = min_leaf
        self.max_depth = max_depth

    def __repr__(self):
        return json.dumps(self.tree, cls=NpEncoder)

    def learn_recursive(self, p_x, p_y, feature_names, level=0):
        if (
                np.var(p_x, axis=0).sum() == 0
                or p_y.size <= self.min_leaf
                or np.var(p_y) == 0
                or level >= self.max_depth
        ):
            return {"label": np.mean(p_y)}

        for (split_value, best_var_reduction), best_feature_index in sorted(
                [(variance_split(p_x[:, i], p_y), i) for i in range(p_x.shape[1])],
                key=lambda _x: _x[0][1],
                reverse=True):

            if not best_var_reduction:
                return {"label": np.mean(p_y)}

            l_mask = p_x[:, best_feature_index] < split_value
            if len(p_x[l_mask]) < self.min_leaf or len(p_x[~l_mask]) < self.min_leaf:
                continue

        curr_feature = feature_names[best_feature_index]
        del feature_names[best_feature_index]
        p_x = np.delete(p_x, best_feature_index, 1)

        l_node = self.learn_recursive(p_x[l_mask], p_y[l_mask], feature_names, level + 1)
        be_node = self.learn_recursive(p_x[~l_mask], p_y[~l_mask], feature_names, level + 1)

        return {
            "split_feature": curr_feature,
            "split_value": split_value,
            "<": l_node,
            ">=": be_node,
        }

    def fit(self, p_x, p_y):
        self.train_x = p_x
        self.train_y = p_y
        feature_names = list(range(p_x.shape[1]))
        self.tree = self.learn_recursive(p_x, p_y, feature_names)

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
