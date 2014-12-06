import numpy as np
import os
from sklearn.linear_model import LinearRegression
from draw_pca import _load_json
import matplotlib.pylab as plt


def main():
    json_file_path = '/home/david/git/MIC-XRD-Polymer/python_scripts/' + \
                     'Data/Json_Files/12_samples_json_truncated_to_stra' + \
                     'in_375.JSON'
    pca_dict = _load_json(json_file_path)[0]
    for key, value in pca_dict.iteritems():
        pca_dict[key] = np.asarray(pca_dict[key])

    keys = sorted(pca_dict.keys())
    strain_dir_path = '/home/david/git/MIC-XRD-Polymer/python_scripts' + \
                      '/Data/Load_Strain_and_Crystallinity/'
    strain_dict = _get_strain_dict(strain_dir_path, keys)
    # for key in keys:
    #     print coefficients(pca_dict[key], strain_dict[key])
    model = ProcessLinkageRegression(1, 1, 1)
    X = np.arange(10)
    X = X
    y = X + 20
    print model.fit(X, y)
    # print 'X', X
    # plt.plot(X, 'ko')
    # plt.show()
    # print _get_regression_data(X, Z, 2, 2)
    diff_X = model._transform(X[:])
    int_X = model._integrate(diff_X.copy())
    print diff_X
    print int_X


class ProcessLinkageRegression(LinearRegression):

    def __init__(self, p, d, q, **kwargs):
        self.p = p
        self.q = q
        self.d = d
        self.IC = None
        super(ProcessLinkageRegression, self).__init__(**kwargs)

    def predict(self, X):
        X_transformed = self._

    def fit(self, X, y):
        X_transformed, y_transformed = self._get_regression_data(X, y)
        super(ProcessLinkageRegression, self).fit(X_transformed, y_transformed)
        return X_transformed, y_transformed

    def _transform(self, X):
        X_diff = self._get_diffs_ICs(X)
        self.IC = X_diff[:self.d]
        return self._get_regressors(X_diff[self.d:], self.p)

    def _get_regression_data(self, X_PCA, X_strain):
        X_PCA_diff = self._get_diffs_ICs(X_PCA)
        X_strain_diff = self._get_diffs_ICs(X_strain)
        PCA_regressor = self._get_regressors(X_PCA_diff[self.d:], self.p)
        strain_regressor = self._get_regressors(X_strain_diff[self.d:], self.q)
        self.IC = X_PCA[:self.d]
        order_diff = self.p - self.q
        print 'order_diff', order_diff
        print PCA_regressor.shape
        print strain_regressor.shape
        if order_diff == 0:
            X = np.concatenate((PCA_regressor[:-1],
                                strain_regressor[1:]), axis=-1)
        elif order_diff > 0:
            X = np.concatenate((PCA_regressor[:-1],
                                strain_regressor[1:][order_diff:]), axis=-1)
        else:
            X = np.concatenate((PCA_regressor[:-1][-order_diff:],
                                strain_regressor[1:]), axis=-1)
        cutoff_index = min(self.p, self.q) + abs(order_diff) + self.d
        print 'cutoff_index', cutoff_index
        return X, X_PCA_diff[cutoff_index:]

    def _get_diffs_ICs(self, X):
        X_tmp = X[:]
        for i in range(self.d):
            X_tmp[i:] = self._get_diff_IC(X_tmp[i:])
        return X_tmp

    def _get_diff_IC(self, X):
        X_tmp = X - np.roll(X, 1)
        X_tmp[0] = X[0]
        return X_tmp

    def _integrate(self, X):
        X_tmp = X[:]
        for i in range(self.d - 2, -1, -1):
            X_tmp[i:] = np.cumsum(X[i:])
        return X_tmp

    def _get_regressors(self, X, order):
        X_regressor = X[:order][None]
        for ii in range(1, X.shape[0] - order + 1):
            index = slice(ii, ii + order)
            X_regressor = np.concatenate((X_regressor, X[index][None]))
        return X_regressor


def coefficients(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_, model.score(X, y)


def _get_strain_dict(strain_dir_path, keys):
    strain_files = sorted([f for f in os.listdir(strain_dir_path)])
    strain_dict = {}
    for strain_file, key in zip(strain_files, keys):
        f = open(os.path.join(strain_dir_path, strain_file), 'rb')
        data = f.readlines()
        data_cleaned = [i[:-1] for i in data]
        split_data = [i.split('\t') for i in data_cleaned]
        tmp_data = map(list, zip(*split_data))[-1][1:]
        next_data = np.array([float(i) for i in tmp_data])
        strain_dict[key] = next_data
    return strain_dict





if __name__ == '__main__':
    main()
