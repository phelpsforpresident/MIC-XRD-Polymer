from sklearn.decomposition import RandomizedPCA
from image_dimension_reducer import ImageDimensionReducer
from pymks.tools import draw_PCA
import time

model = RandomizedPCA(n_components=3)
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
#=======
#print 'Load PCA'
#model = PCA(n_components=3)
#print 'Create PCA'
#IDR = ImageDimensionReducer(model=model)
#print 'Load Data'
'''
IDR.load_data('/Users/abhiramkannan/Documents/SAXS_for_DB/', '.tif')
print 'Prep Data'
IDR.prep_data()
print 'Transform Data'
X_PCA = IDR.fit_transform()
normalized_eigenvalues = IDR.model.explained_variance_ratio_
print 'Percent Variance', normalized_eigenvalues
draw_PCA(X_PCA, 12) 
>>>>>>> Stashed changes
'''
