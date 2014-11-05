from sklearn.decomposition import PCA
from image_dimension_reducer import ImageDimensionReducer
from pymks.tools import draw_PCA

model = PCA(n_components=3)
IDR = ImageDimensionReducer(model=model)
IDR.load_data('/home/david/matials_data_temp/', '.tif')
IDR.prep_data()
X_PCA = IDR.fit_transform()
draw_PCA(X_PCA, 1)
