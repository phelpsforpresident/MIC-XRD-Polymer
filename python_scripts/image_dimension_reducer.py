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
        self.data = None
        self.reduced_data = None
        self.n_datasets = 0
        self.n_datasets_names = []

    def load_images(self, directory_path=None, file_type=None):
        '''
        load data from a directory

        Args:
            directory_path: path to directory containing image files.
            file_type: provides an option to specify the type of file to be
                loaded.
        '''
        if directory_path is None:
            raise RuntimeError("directory_path not specified")
        if file_type is None:
            raise RuntimeError("file_type not specified")
        self._load_images(directory_path, file_type)

    def _load_images(self, directory, file_type):
        '''
        Helper funciton to load image files.
        '''
        file_names = sorted(os.listdir(directory))
        file_index = 0
        while not file_names[file_index].endswith(file_type):
            file_index = file_index + 1
        im = skio.imread(os.path.join(directory, file_names[0]))
        raw_data = im[None]
        if self.data is None:
            self.data = raw_data
        else:
            self._check_dimensions(raw_data)
        self.n_datasets_names.append(file_names[file_index])
        self.n_datasets = self.n_datasets + 1
        for file_name in file_names[file_index:]:
            if file_name.endswith(file_type):
                im = skio.imread(os.path.join(directory, file_name))
                raw_data = np.concatenate((raw_data, im[None]))
                self.n_datasets = self.n_datasets + 1
                self.n_datasets_names.append(file_name)
        self.data = np.concatenate((self.data, raw_data))

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
        formated_data = self._format_data(self.data)
        self.reduced_data = self.model.fit_transform(formated_data)
        return self.reduced_data

    def remove_data(self):
        '''
        removes all data from ImageDimensionReducer
        '''
        self.data = None
        self.reduced_data = None
        self.n_datasets = 0
        self.n_datasets_names = []

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
            index = range(self.n_datasets, X.shape[0])
            new_name = index
        else:
            index = range(0, X.shape[0])
            new_name = [name + str(x) for x in index]
        self.n_datasets_names = self.n_datasets_names + new_name
        if self.data is None:
            self.data = X
        else:
            self._check_dimensions(X)
            self.data = np.concatnate(self.data, X)

    def _check_dimensions(self, X):
        if self.data[1:] != X.shape[:1]:
            raise RuntimeError("Array sizes don't match")
