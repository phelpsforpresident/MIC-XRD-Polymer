import numpy as np
from image_dimension_reducer import ImageDimensionReducer


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
        log_data = np.log(sliced_data + 10)
        #slice_max = np.amax(log_data, axis=tuple(range(1, log_data.ndim)))
        #slice_median = np.median(log_data,
        #                         axis=tuple(range(1, log_data.ndim)))
        slice_mean = np.mean(log_data, axis=tuple(range(1, log_data.ndim)))
        #log_norm_data = log_data * 1. / slice_max[:, None, None]
        #log_norm_data = log_data * 1. / slice_median[:, None, None]
        log_norm_data = log_data * 1. / slice_mean[:, None, None]
        self.manipulated_data = log_norm_data

    def fit_transform(self):
        '''
        This function leverages the model's fit_transform function to create a
        low dimension representation of the data.

        Returns: Reduced dimension representation of the raw_data.
        '''
        formated_data = self._format_data(self.manipulated_data)
        if hasattr(self, 'dataset_lables'):
            self.reduced_data = self.reducer.fit_transform(formated_data,
                                                           self.dataset_lables)
        else:
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
        thumbnail_data = self.manipulated_data
        self._make_thumbnails(thumbnail_data=thumbnail_data,
                              thumbnail_path=thumbnail_path,
                              thumbnail_size=thumbnail_size,
                              thumbnail_type=thumbnail_type)
