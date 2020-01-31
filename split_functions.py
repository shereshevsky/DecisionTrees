import numpy as np


def variance_split(p_x: np.array, p_y: np.array, min_leaf: int):
    """
    Calculate variance for all the possible split positions
    :param p_x: np.array of feature values
    :param p_y: np.array of target values
    :param min_leaf: minimal number values on leaf to avoid invalid splits
    :return:
    """
    # create 2d array of single feature and target values and sort by feature values.
    # 0 - feature, 1 - target
    arr = np.array(sorted(np.vstack([p_x, p_y]).T, key=lambda v: v[0])).T
    # initial variance w/o any split
    init_var = np.var(p_y) / p_y.size
    # generate variance for all possible feature splits
    all_vars = []
    for i in range(p_y.size):
        # don't split < min_leaf size, don't split if the next value == current value
        if i < min_leaf or i > (p_y.size - min_leaf) or arr[0][i] == arr[0][i-1]:
            all_vars.append(1000)
        else:
            all_vars.append(
                (np.var(arr[1][:i]) / i + np.var(arr[1][i:]) / (p_y.size - i)) / p_y.size
            )

    index_of_lowest_var = np.where(np.array(all_vars) == np.min(np.array(all_vars)))[0][0]

    found_var = all_vars[index_of_lowest_var]
    split_value = arr[0][index_of_lowest_var]

    if found_var >= init_var:
        return -1, 0
    else:
        return split_value, init_var - found_var
