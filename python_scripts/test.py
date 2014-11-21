import time
from sklearn.decomposition import KernelPCA
from image_manipulator_dimension_reducer \
    import ImageManipulatorDimensionReducer
from pymks.tools import draw_PCA

#reducer = LDA(n_components=3, priors=3)
reducer = KernelPCA(n_components=3)

IMDR = ImageManipulatorDimensionReducer(reducer=reducer)

t_start = time.time()

IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_1/',
#IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAXS_01')
print 'Loaded data in', time.time() - t_start, 'secs'

IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_2/',
#IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAXS_02')
print 'Loaded data in', time.time() - t_start, 'secs'


IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_3/',
#IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAXS_03')
print 'Loaded data in', time.time() - t_start, 'secs'

IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_4/',
#IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAXS_04')
print 'Loaded data in', time.time() - t_start, 'secs'

IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_5/',
#IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAXS_05')
print 'Loaded data in', time.time() - t_start, 'secs'

IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_6/',
#IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAXS_06')
print 'Loaded data in', time.time() - t_start, 'secs'

IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_7/',
#IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAXS_07')
print 'Loaded data in', time.time() - t_start, 'secs'

IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_8/',
#IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAXS_08')
print 'Loaded data in', time.time() - t_start, 'secs'

## Sample 9 ruptured too early

#IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_9/',
# #IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
#                  file_type='.tif', dataset_name='SAXS_9')
# print 'Loaded data in', time.time() - t_start, 'secs'

IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_10/',
#IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAXS_10')
print 'Loaded data in', time.time() - t_start, 'secs'

IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_11/',
#IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAXS_11')
print 'Loaded data in', time.time() - t_start, 'secs'

IMDR.load_images('/Users/abhiramkannan/Documents/SAXS_for_DB_12/',
#IMDR.load_images(directory_path='/home/david/Pictures/SAXS_for_DB/',
                 file_type='.tif', dataset_name='SAXS_12')
print 'Loaded data in', time.time() - t_start, 'secs'

t_start = time.time()
IMDR.manipulate(new_slice=(slice(None), slice(290, 745), slice(217, 612))) #slice for running PCA
#IMDR.manipulate(new_slice=(slice(None), slice(290, 728), slice(237, 612))) #slice for running 2 point statistics via FFT
print 'Manipulated data in', time.time() - t_start, 'secs'

t_start = time.time()
IMDR.make_data_labels()

t_start = time.time()
X_PCA = IMDR.fit_transform()
print 'Fit PCA in ', time.time() - t_start, 'secs'

t_start = time.time()
IMDR.reduced_data_to_json(file_name='tmp_json.JSON',
                          file_path='/Users/abhiramkannan/Desktop/')
print 'Made JSON in ', time.time() - t_start, 'secs'


# t_start = time.time()
# IMDR.make_thumbnails(
#     thumbnail_path='/Users/abhiramkannan/Documents/SAXS_for_DB_Remaining/thumbnails/',
#     thumbnail_size=(75, 75), thumbnail_type='.png')
# print 'Made thumbnails in ', time.time() - t_start, 'secs'

print 'data shape', IMDR.manipulated_data.shape
print 'X_PCA shape', X_PCA.shape
draw_PCA(X_PCA, 12)

