import numpy as np
from TreeEnsemble import TreeEnsemble

if __name__ == "__main__":
    from sklearn.datasets import load_boston

    np.random.seed(42)

    boston = load_boston()
    boston_y = boston.target
    boston_X = boston.data
    #
    # X = np.array([[1, 5, 1],
    #               [4, 5, 3],
    #               [3, 5, 4],
    #               [2, 6, 2],
    #               [1, 6, 2]])
    # y = np.array([1, 1, 0, 0, 1])
    # forest = TreeEnsemble(X, y, n_trees=2, sample_sz=4, min_leaf=2)
    #
    # print(forest.trees[0])
    # print(forest.trees[1])
    #
    # # print([(t.gt_x, t.gt_y) for t in forest.trees])
    #
    # print(forest.predict(np.array([[0, 5, 1],
    #                                [2, 3, 2],
    #                                [1, 2, 3],
    #                                [1, 5, 4],
    #                                [1, 5, 1],
    #                                [4, 5, 3],
    #                                [3, 5, 4],
    #                                [2, 6, 2],
    #                                [1, 6, 2]
    #                                ])))

    forest_boston = TreeEnsemble(
        boston_X, boston_y, n_trees=10, sample_sz=500, min_leaf=5
    )

    print(forest_boston.predict(boston_X[88:99]))

    print([t.mse() for t in forest_boston.trees])

    print(forest_boston.oob_mse())