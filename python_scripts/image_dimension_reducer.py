import numpy as np
import os
import skimage.io as skio
#from libtiff import TIFF


class ImageDimensionReducer(object):
    """
    This `ImagePCA` uses Principle Component Analysis (PCA) to objectively
    find the differences between images.
    """

    def __init__(self, model=None):
        '''
        Create an instance of ImageDimensionReducer.
        '''
        self.model = model
        if self.model is None:
            raise RuntimeError("Must specify model")
        self.data = {}
        self.preped_data = None
        self.n_t_steps = 0
        self.n_microstructures = 0

    def Load_data(self, data_file=None, data_name=None):
        if data_name is None:
            raise RuntimeError("Must specify data_name")
        if data_file is None:
            raise RuntimeError("Must specify data_file")
        sample_data = self._get_data(data_name)
        self.data[data_name] = sample_data

    def load_data(self, folder, file_type):
        k = 0
        file_names = sorted(os.listdir(folder))
        for file_name in file_names:
            if k == (k + 1) / 2:
                if file_name.endswith(file_type):
                    imarray = skio.imread(os.path.join(folder, file_name))
                    imarray = np.log(imarray + 0.1)
                    imarray = imarray / np.max(imarray)
                    imarray = imarray[:600, 400:]
                    if k == 0:
                        raw_data = imarray[None]
                        k = k + 1
                    else:
                        raw_data = np.concatenate((raw_data, imarray[None]))
            else:
                k = k + 1
        data = self._format_data(raw_data)
        self.data[folder] = data
        self.n_microstructures = self.n_microstructures + 1

    def prep_data(self):
        size = self.data[self.data.keys()[0]].shape
        l = len(self.data)
        self.n_t_steps = size[0]
        size = (l * self.n_t_steps,) + size[1:]
        preped_data = np.zeros(size)
        i = 0
        for key, value in self.data.iteritems():
            preped_data[i * self.n_t_steps:(i + 1) * self.n_t_steps] = value
        self.preped_data = preped_data

    def _format_data(self, raw_data):
        size = np.array(raw_data.shape)
        new_size = (size[0], np.prod(size[1:]))
        return raw_data.reshape(new_size)

    def fit_transform(self):
        return self.model.fit_transform(self.preped_data)

    def remove_data(self):
        self.data = {}
        self.preped_data = None
        self.n_t_steps = 0
        self.n_microstructures = 0
