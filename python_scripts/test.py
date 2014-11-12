from sklearn.decomposition import KernelPCA
from image_dimension_reducer import ImageDimensionReducer
from pymks.tools import draw_PCA
import time
import matplotlib.pylab as plt

model = KernelPCA(n_components=3)
IDR = ImageDimensionReducer(model=model)
t_start = time.time()
#IDR.load_data('/Users/abhiramkannan/Documents/SAXS_for_DB/', '.tif')
IDR.load_data('/home/david/Pictures/Data_for_ME_8833/', '.tif')
print 'Loaded Data in', time.time() - t_start, 'sec'
t_start = time.time()
IDR.prep_data()
print 'Prep Data in', time.time() - t_start, 'sec'
t_start = time.time()
X_PCA = IDR.fit_transform()
print 'Fit PCA in ', time.time() - t_start, 'sec'
draw_PCA(X_PCA, 8)

eigen_values = IDR.model.lambdas_
plt.plot(eigen_values, 'ko-')
plt.title("Eigen Values")
plt.show()
