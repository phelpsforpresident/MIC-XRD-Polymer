import numpy as np
import os
import skimage.io as skio


class ImageDimensionReducer(object):

    """
    This `ImageDimensionReducer` uses takes in a dimensionality reduction
    class from sklearn, and creates a reduced representation of the images.
    """

    def __init__(self, model=None):
        '''
        Create an instance of ImageDimensionReducer.

        Args:
            model: dimensionality reduction class from sklearn. The class must
                have a 'fit_transform' method.
        '''
        self.model = model
        if self.model is None:
            raise RuntimeError("Must specify model")
        self.data = {}
        self.reduced_data = {}
        self.n_images = 0
        self.n_arrays = 0
        self.n_datasets = 0

    def load_images(self, directory_path=None, file_type=None):
        '''
        load data from a directory

        Args:
            directory_path: path to directory containing image files.
            file_type: provides an option to specify the type of file to be
                loaded.
        '''
        if directory_path is None:
            raise RuntimeError("Must specify directory_path")
        if file_type is None:
            file_type = '.*'
        self._load_images(directory_path, file_type)

    def _load_images(self, directory, file_type):
        '''
        Helper funciton to load image files.
        '''
        file_names = sorted(os.listdir(directory))
        im = skio.imread(os.path.join(directory, file_names[0]))
        raw_data = im[None]
        self.data[file_names[0]] = raw_data
        self.n_datasets = self.n_datasets + 1
        for file_name in file_names[1:]:
            if file_name.endswith(file_type):
                im = skio.imread(os.path.join(directory, file_name))
                raw_data = np.concatenate((raw_data, im[None]))
                self.n_images = self.n_images + 1
        self.data[file_name] = raw_data
        self.n_datasets = self.n_datasets + self.n_images

    def _manipulate_data(self, im):
        '''
        Helper function to manipulate arrays representing images.
        '''
        im_log = np.log(im + 1)
        im_log_norm = im_log / np.max(im_log)
        return im_log_norm[:600, 400:]

    def _prep_data(self):
        '''
        Helper function that pulls data out of dictionary and into an array.
        '''
        size = self.data[self.data.keys()[0]].shape
        size = (self.n_datasets,) + (np.prod(size[1:]),)
        preped_data = np.zeros(size)
        for key, value in self.data.iteritems():
            formated_data = self._format_data(value)
            preped_data[:self.n_datasets] = formated_data
        return preped_data

    def _format_data(self, raw_data):
        '''
        Changes the size of the data to mee
        '''
        size = np.array(raw_data.shape)
        new_size = (size[0], np.prod(size[1:]))
        return raw_data.reshape(new_size)

    def fit_transform(self):
        '''
        This function leverages the model's fit_transform function to reduce
        the number of dimensions.
        '''
        preped_data = self._prep_data()
        self.reduced_data = self.model.fit_transform(preped_data)
        return self.reduced_data

    def remove_data(self):
        '''
        removes all data from ImageDimensionReducer
        '''
        self.data = {}
        self.reduced_data = {}
        self.n_images = 0
        self.n_arrays = 0
        self.n_datasets = 0

    def save_to_json(self):
        '''
        Saves data to json file.
        '''
        if self.data == {}:
            raise RuntimeError("No data to save.")
        raise NotImplementedError("Not needed yet.")

    def load_array(self, X, name=None):
        '''
        Loads an array as data.
        '''
        if name is None:
            name = str(self.n_datasets + 1)
        self.data[name] = X[None]
