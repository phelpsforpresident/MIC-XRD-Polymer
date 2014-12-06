import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json


def draw_PCA_from_json(path, title):
    data = _load_json(path)
    n_components = len(data[0].values()[0][0])
    if n_components == 2:
        _draw_PCA_2D_from_dict(data[0], title)
    elif n_components == 3:
        _draw_PCA_3D_from_dict(data[0], title)
    else:
        raise RuntimeError('n_components must be either 2 or 3')


def _load_json(path):
    data = []
    with open(path, 'rb') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def draw_PCA(X, n_sets):
    size = np.array(X.shape)
    if size[-1] == 2:
        _draw_PCA_2D(X, n_sets)
    elif size[-1] == 3:
        _draw_PCA_3D(X, n_sets)
    else:
        raise RuntimeError("n_components must be 2 or 3.")


def _get_PCA_color_list(n_sets):
    color_list = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
                  '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
                  '#cab2d6', '#6a3d9a', '#1a1a1a', '#b15928']
    return color_list[:n_sets]


def _draw_PCA_2D(X, n_sets):
    color_list = _get_PCA_color_list(n_sets)
    sets = np.array(X.shape)[0] / n_sets
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('PC 1', fontsize=15)
    ax.set_ylabel('PC 2', fontsize=15)
    ax.set_xticks(())
    ax.set_yticks(())
    for n in range(n_sets):
        ax.scatter(X[n * sets:(n + 1) * sets, 0],
                   X[n * sets:(n + 1) * sets, 1],
                   color=color_list[n])


def _draw_PCA_3D(X, n_sets):
    color_list = _get_PCA_color_list(n_sets)
    sets = np.array(X.shape)[0] / n_sets
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('PC 1', fontsize=10)
    ax.set_ylabel('PC 2', fontsize=10)
    ax.set_zlabel('PC 3', fontsize=10)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())
    for n in range(n_sets):
        ax.scatter(X[n * sets:(n + 1) * sets, 0],
                   X[n * sets:(n + 1) * sets, 1],
                   X[n * sets:(n + 1) * sets, 2],
                   color=color_list[n])


def _draw_PCA_3D_from_dict(X_dict, title):
    color_list = _get_PCA_color_list(len(X_dict))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('PC 1', fontsize=10)
    ax.set_ylabel('PC 2', fontsize=10)
    ax.set_zlabel('PC 3', fontsize=10)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())
    color_index = 0
    keys = sorted(X_dict.keys())
    for color, key in zip(color_list, keys):
        x_pt = []
        y_pt = []
        z_pt = []
        value = X_dict[key]
        for point in value:
            x_pt.append(point[0])
            y_pt.append(point[1])
            z_pt.append(point[2])
        ax.scatter(x_pt, y_pt, z_pt, color=color, label=key)
        color_index += 1
    plt.legend(keys)
    plt.title(title)
    plt.show()


def _draw_PCA_2D_from_dict(X_dict, title):
    color_list = _get_PCA_color_list(len(X_dict))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('PC 1', fontsize=15)
    ax.set_ylabel('PC 2', fontsize=15)
    ax.set_xticks(())
    ax.set_yticks(())
    keys = sorted(X_dict.keys())
    for key, color in zip(keys, color_list):
        x_pt = []
        y_pt = []
        value = X_dict[key]
        for point in value:
            x_pt.append(point[0])
            y_pt.append(point[1])
        ax.scatter(x_pt, y_pt, color=color, label=key)
    plt.legend(keys)
    plt.title(title)
    plt.show()
