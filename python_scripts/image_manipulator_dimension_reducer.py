import numpy as np
import os
from image_dimension_reducer import ImageDimensionReducer
from PIL import Image


class ImageManipulatorDimensionReducer(ImageDimensionReducer):

    '''
    This class allows for the images to be manipulated before dimensionality
    reduction takes place. This only work for 2D images.
    '''

    def manipulate(self, new_slice=slice(None)):
        """
        Manipulates the original data by slicing, taking the log, and then
        normalizing the array.
        """
        sliced_data = self.data[new_slice]
        log_data = np.log(sliced_data + 1)
        slice_max = np.amax(log_data, axis=tuple(range(1, log_data.ndim)))
        log_norm_data = log_data * 1. / slice_max[:, None, None]
        self.manipulated_data = log_norm_data

    def fit_transform(self):
        '''
        This function leverages the model's fit_transform function to create a
        low dimension representation of the data.

        Returns: Reduced dimension representation of the raw_data.
        '''
        formated_data = self._format_data(self.manipulated_data)
        self.reduced_data = self.reducer.fit_transform(formated_data)
        return self.reduced_data

    def make_thumbnails(self, thumbnail_path=None,
                        thumbnail_size=(200, 200), thumbnail_type='.png'):
        '''
        Creates thumbnails of the images.

        Ags:
            thumbnail_path: path to directory where thumbnails are exported.
            thumbnail_size: size of thumbnails
            thumbnail_type: file type of the thumbnails
        '''
        if thumbnail_path is None:
            raise RuntimeError('thumbnail_path not specified')
        try:
            os.stat(thumbnail_path)
        except:
            os.mkdir(thumbnail_path)
        for index in range(self.n_samples):
            im = Image.fromarray((255 * self.manipulated_data[index]).
                                 astype(np.uint8))
            im.save(os.path.join(thumbnail_path,
                                 self.n_samples_names[index] + thumbnail_type))
