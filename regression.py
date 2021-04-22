import csv
import math
import random

import numpy as np
from matplotlib import pyplot as plt


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    INPUT:
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    reader = csv.reader(open(filename), delimiter=',')
    next(reader)  # removes the header

    for line in reader:
        dataset.append(line)

    dataset = np.array(dataset, np.float64)
    dataset = np.delete(dataset, 0, 1)
    return dataset


def print_stats(dataset, col):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on.
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    n = dataset.shape[0]
    mean = 0
    std = 0

    for row in dataset:
        mean += row[col]

    mean = mean / n

    for row in dataset:
        std += ((row[col] - mean) ** 2)

    std = std / (n - 1)
    std = math.sqrt(std)

    mean = '%.2f' % mean
    std = '%.2f' % std

    print(n)
    print(mean)
    print(std)
    pass


def calc_mse(dataset, cols, betas, partial=None):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model or the partial derivative of mse without the coefficient
    """
    total = 0
    n = dataset.shape[0]
    num_cols = len(cols)

    if len(np.shape(dataset)) > 1:
        for row in range(n):
            sub_total = betas[0]
            for col in range(num_cols):
                sub_total += betas[col + 1] * dataset[row, cols[col]]
            if partial is None:
                total += (sub_total - dataset[row, 0]) ** 2
            else:
                if partial == 0:
                    total += sub_total - dataset[row, 0]
                else:
                    total += (sub_total - dataset[row, 0]) * dataset[row, cols[partial - 1]]
    else:
        sub_total = betas[0]
        for col in range(num_cols):
            sub_total += betas[col + 1] * dataset[cols[col]]
        if partial == 0:
            total += sub_total - dataset[0]
        else:
            total += (sub_total - dataset[0]) * dataset[cols[partial - 1]]
    return total


def regression(dataset, cols, betas):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    n = dataset.shape[0]

    mse = calc_mse(dataset, cols, betas) / int(n)

    return mse


def gradient_descent(dataset, cols, betas):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    n = dataset.shape[0]
    grads = []

    for x in range(len(betas)):
        grads.append(2 * (calc_mse(dataset, cols, betas, partial=x) / int(n)))

    return np.array(grads)


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    arr = list.copy(betas)
    for t in range(1, T + 1):
        st = ""
        grads = gradient_descent(dataset, cols, betas)
        for b in range(len(betas)):
            arr[b] = betas[b] - eta * grads[b]
            st += '{:.2f} '.format(round(arr[b], 2))
        print('{} {:.2f} {}'.format(t, round(regression(dataset, cols, arr), 2), st[:-1]))
        betas = list.copy(arr)
    pass


def compute_betas(dataset, cols):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    x = np.c_[np.ones(len(dataset[:, 0])), np.array(dataset[:, cols])]
    y = dataset[:, 0]
    betas = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
    mse = regression(dataset, cols, betas)
    return mse, *betas


def predict(dataset, cols, features):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)
    result = betas[1]
    for x in range(len(features)):
        result += betas[x + 2] * features[x]
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    INPUT:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    linear = []
    quadratic = []
    for x in X:
        z = np.random.normal(0, sigma)
        line = betas[0] + (betas[1] * x[0]) + z
        linear.append([line, x[0]])
        quad = alphas[0] + (alphas[1] * (x[0] ** 2)) + z
        quadratic.append([quad, x[0]])

    return np.array(linear), np.array(quadratic)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    X = np.array([random.randint(-100, 100) for _ in range(1000)]).reshape(1000, 1)
    sigmas = [.0001, .001, .01, .1, 1, 10, 100, 1000, 10000, 100000]
    betas = [random.randint(1, 10) for _ in range(2)]
    alphas = [random.randint(1, 10) for _ in range(2)]

    fig, ax = plt.subplots(figsize=(8, 8))

    plt.xlabel('Sigma')
    plt.ylabel('MSEs')

    ax.set_title('Bodyfat')
    ax.set_xscale('log')
    ax.set_yscale('log')

    y_linear = []
    y_quadratic = []

    for sigma in sigmas:
        linear_dataset, quadratic_dataset = synthetic_datasets(betas, alphas, X, sigma)
        linear_mse = compute_betas(linear_dataset, cols=[1])[0]
        y_linear.append(linear_mse)
        quadratic_mse = compute_betas(quadratic_dataset, cols=[1])[0]
        y_quadratic.append(quadratic_mse)

    red_points, = ax.plot(sigmas, y_linear, '-o', label="Linear Dataset", color='tab:red')
    cyan_points, = ax.plot(sigmas, y_quadratic, '-o', label="Quadratic Dataset", color='tab:cyan')

    plt.legend([red_points, cyan_points], ['Linear', 'Quadratic'])


if __name__ == '__main__':
    plot_mse()
