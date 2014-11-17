import numpy as np
import os
import json
import skimage.io as skio
from PIL import Image


class ImageDimensionReducer(object):

    """
    This `ImageDimensionReducer` uses takes in a dimensionality reduction
    class from sklearn, and creates a reduced representation of the images.
    """

    def __init__(self, model=None, thumbnails_size=(75, 75)):
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
        self.n_samples = 0
        self.n_samples_names = []
        self.thumbnails_size = thumbnails_size
        self.datasets = {}

    def load_images(self, directory_path=None,
                    file_type=None, dataset_name=None):
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
        self._load_images(directory_path, file_type, dataset_name)

    def _load_images(self, directory, file_type, dataset_name=None):
        '''
        Helper function to load image files.
        '''
        file_names = sorted(os.listdir(directory))
        file_index = 0
        dataset_start_index = self.n_samples
        while not file_names[file_index].endswith(file_type):
            file_index = file_index + 1
        im = skio.imread(os.path.join(directory, file_names[0]))
        raw_data = im[None]
        if self.data is None:
            self.data = raw_data
            final_index = 1
        else:
            self._check_dimensions(raw_data)
            final_index = 0
        self.n_samples_names.append(file_names[file_index][:-4])
        self.n_samples = self.n_samples + 1
        for file_name in file_names[file_index + 1:]:
            if file_name.endswith(file_type):
                im = skio.imread(os.path.join(directory, file_name))
                raw_data = np.concatenate((raw_data, im[None]))
                self.n_samples = self.n_samples + 1
                self.n_samples_names.append(file_name[:-4])
        self.data = np.concatenate((self.data, raw_data[final_index:]))
        if dataset_name is None:
            dataset_name = file_names[0][:-9]
        self.datasets[dataset_name] = slice(dataset_start_index,
                                            self.n_samples)

    def _prep_data(self):
        '''
        Helper function that pulls data out of dictionary and into an array.
        '''
        size = self.data[self.data.keys()[0]].shape
        size = (self.n_samples,) + (np.prod(size[1:]),)
        preped_data = np.zeros(size)
        for key, value in self.data.iteritems():
            formated_data = self._format_data(value)
            preped_data[:self.n_samples] = formated_data
        return preped_data

    def _format_data(self, raw_data):
        '''
        Changes the size of the data to meet the format required to do
        dimensionality reduction.
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

    def clear_data(self):
        '''
        removes all data from ImageDimensionReducer
        '''
        self.data = None
        self.reduced_data = None
        self.n_samples = 0
        self.n_samples_names = []

    def reduced_data_to_json(self, file_name=None, file_path=None):
        '''
        Saves data to json file.
        '''
        if file_name is None:
            raise RuntimeError("file_name not specified")
        if file_path is None:
            raise RuntimeError("file_path not specified")
        if self.data is None:
            raise RuntimeError("No data.")
        if self.reduced_data is None:
            raise RuntimeError("No reduced data.")
        reduced_data_dict = {}
        for key, value in self.datasets.iteritems():
            reduced_data_dict[key] = self.reduced_data[value].tolist()
        print file_name
        with open(os.path.join(file_path, file_name), 'w') as json_file:
            json.dump(reduced_data_dict, json_file)

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
            im = Image.fromarray(self.data[index].astype(np.uint8))
            im.save(os.path.join(thumbnail_path,
                    self.n_samples_names[index] + thumbnail_type))

    def load_array(self, X, dataset_name=None):
        '''
        Loads an array as data.
        '''
        index = range(self.n_samples, X.shape[0])
        if dataset_name is None:
            dataset_name = str(len(self.datasets))
        new_name = [dataset_name + str(x) for x in index]
        dataset_start_index = len(self.n_samples)
        if self.data is None:
            self.data = X
        else:
            self._check_dimensions(X)
            self.data = np.concatnate(self.data, X)
        self.n_samples_names = self.n_samples_names + new_name
        self.datasets[dataset_name] = slice(dataset_start_index,
                                            len(self.n_samples_names))

    def _check_dimensions(self, X):
        if self.data.shape[1:] != X.shape[:1]:
            raise RuntimeError("Array sizes don't match")
