import numpy as np
import os
import matplotlib.pylab as plt
from PIL import Image
from dropbox_path import dropbox_path


class ImageDimensionReducer(object):

    """
    This `ImagePCA` uses Principle Component Analysis (PCA) to objectively
    find the differences between images.
    """

    def __init__(self, model=None):
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
        sample_data = self._get_data(file_name)
        self.data[data_name] = sample_data

    def load_data(self, folder, file_type):
        directory_path = os.path.join(dropbox_path(), folder)
        k = 0
        for i in os.listdir(directory_path):
            if i.endswith(file_type):
                script_dir = os.path.dirname(os.getcwd())
                im = Image.open(os.path.join(directory_path, i))
                im = im.convert("L")
                imarray = np.array(im)
                if imarray.shape == (2832, 4256) and k < 10:
                    if k == 0:
                        raw_data = imarray[None]
                        k = 1
                    else:
                        raw_data = np.concatenate((raw_data, imarray[None]))
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
        self.n_microstructures =0

