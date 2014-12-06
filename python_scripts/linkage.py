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
    d = 0
    model = ProcessLinkageRegression(1, d, 1)
    X = np.arange(10)
    strain = X ** 2 + 20
    model.fit(X.copy(), strain.copy())
    y_pred = model.predict(X.copy(), strain.copy(), X[:d])
    plt.plot(X, 'ko')
    plt.plot(y_pred, 'r-')
    plt.show()
    print model._get_regression_data(X, strain, 2, 2)
    print model._get_regressors(X, 3)



class ProcessLinkageRegression(LinearRegression):

    def __init__(self, p, d, q, **kwargs):
        self.p = p
        self.q = q
        self.d = d
        self.IC = None
        self.gap_values = None
        super(ProcessLinkageRegression, self).__init__(**kwargs)

    def predict(self, X, strain, initial_conditions=np.array([])):
        if initial_conditions.shape[0] != self.d:
            'number of initial_conditions is incorrect'
        self.IC = initial_conditions
        X_pred_trans, y_transformed = self._get_regression_data(X, strain)
        # print 'X_predict', X_pred_trans
        y_pred_trans = super(ProcessLinkageRegression,
                             self).predict(X_pred_trans)
        # print 'y_predict', y_pred_trans
        print 'self.IC', self.IC
        return self._integrate(np.append(self.gap_values, y_pred_trans))

    def fit(self, X, strain):
        print 'X', X
        print 'strain', strain
        X_transformed, y_transformed = self._get_regression_data(X, strain)
        print X_transformed.shape
        print y_transformed.shape
        print 'X_fit', X_transformed
        print 'y_fit', y_transformed
        super(ProcessLinkageRegression, self).fit(X_transformed, y_transformed)

    def _predict_transform(self, X):
        X_regressor = self._transform(X, self.p)
        return X_regressor

    def _transform(self, X, order):
        X_diff = self._get_diffs_ICs(X)
        return self._get_regressors(X_diff[self.d:], order)

    def _get_regression_data(self, X_PCA, X_strain):
        # PCA_regressor = self._transform(X_PCA, self.p)
        strain_regressor = self._transform(X_strain, self.q)
        X_PCA_diff = self._get_diffs_ICs(X_PCA)
        PCA_regressor = self._get_regressors(X_PCA_diff[self.d:], self.p)
        # self.IC = X_PCA[:self.d]
        order_diff = self.p - self.q
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
        # cutoff_index = abs(order_diff) + self.d
        self.gap_values = X_PCA_diff[:cutoff_index]
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
        X_tmp = X.copy()
        print X_tmp
        print range(self.d - 1, -1, -1)
        for i in range(self.d - 1, -1, -1):
            print 'self.IC[i]', self.IC[i]
            X_tmp = np.cumsum(np.append(self.IC[i], X_tmp))
            print X_tmp
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
