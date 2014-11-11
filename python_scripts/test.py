from sklearn.decomposition import KernelPCA
from image_dimension_reducer import ImageDimensionReducer
from pymks.tools import draw_PCA
import time

model = KernelPCA(n_components=3)
IDR = ImageDimensionReducer(model=model)
t_start = time.time()
IDR.load_data('/Users/abhiramkannan/Documents/SAXS_for_DB/', '.tif')
print 'Loaded Data in', time.time() - t_start, 'sec'
t_start = time.time()
IDR.prep_data()
print 'Prep Data in', time.time() - t_start, 'sec'
t_start = time.time()
X_PCA = IDR.fit_transform()
print 'Fit PCA in ', time.time() - t_start, 'sec'
draw_PCA(X_PCA, 12)
