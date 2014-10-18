from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from image_dimension_reducer import ImageDimensionReducer
from pymks.tools import draw_PCA

model = PCA(n_components=2)
IDR = ImageDimensionReducer(model=model)
IDR.load_data('Brough Family Pictures', '.JPG')
IDR.prep_data()
X_PCA = IDR.fit_transform()

model = MDS(n_components=2)
IDR.model = model
X_MDS = IDR.fit_transform()

draw_PCA(X_PCA, 2)
draw_PCA(X_MDS, 2)




