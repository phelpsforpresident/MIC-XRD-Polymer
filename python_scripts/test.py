import time
from sklearn.decomposition import KernelPCA
from image_manipulator_dimension_reducer \
    import ImageManipulatorDimensionReducer
from pymks.tools import draw_PCA

reducer = KernelPCA(n_components=3)
IMDR = ImageManipulatorDimensionReducer(reducer=reducer)

t_start = time.time()
#IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB/', '.tif')
IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAX_2')
print 'Loaded data in', time.time() - t_start, 'secs'

t_start = time.time()
IMDR.manipulate(new_slice=(slice(None), slice(360, None), slice(None, 600)))
print 'Manipulated data in', time.time() - t_start, 'secs'

t_start = time.time()
X_PCA = IMDR.fit_transform()
print 'Fit PCA in ', time.time() - t_start, 'secs'

t_start = time.time()
#IMDR.reduced_data_to_json(file_name='tmp_json.JSON',
#                          file_path='/home/david/Desktop/')
print 'Made JSON in ', time.time() - t_start, 'secs'


t_start = time.time()
IMDR.make_thumbnails(
    thumbnail_path='/home/david/Pictures/SAXS_for_DB/thumbnails/',
    thumbnail_size=(75, 75), thumbnail_type='.png')
print 'Made thumbnails in ', time.time() - t_start, 'secs'

draw_PCA(X_PCA, 1)
