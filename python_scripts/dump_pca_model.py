import skimage.io as io
from os import listdir
import numpy as np
from sklearn.decomposition import RandomizedPCA
import cPickle as pickle

root_path = '/Users/abhiramkannan/Documents/'
directories = ['SAXS_for_DB_1/', 'SAXS_for_DB_2/', 'SAXS_for_DB_3/',
               'SAXS_for_DB_4/', 'SAXS_for_DB_5/', 'SAXS_for_DB_6/',
               'SAXS_for_DB_7/', 'SAXS_for_DB_8/', 'SAXS_for_DB_9/',
               'SAXS_for_DB_10/', 'SAXS_for_DB_11/', 'SAXS_for_DB_12/']

file_paths = [d + i for d in directories for i in listdir(root_path + d)
              if i[-3:] == 'png']

X = np.concatenate([io.imread(root_path + p)[None] for p in file_paths])
X_reshaped = np.log(X.reshape(X.shape[0], X[0].size) + 10)
X_reshaped /= np.mean(X_reshaped, axis=-1)[:, None]
rpca = RandomizedPCA(n_components=7)
y = rpca.fit_transform(X_reshaped)
pickle.dump((rpca, y, file_paths), open(root_path + 'rpca.pkl', 'wb'))
