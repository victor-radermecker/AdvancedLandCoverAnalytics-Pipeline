import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
import os


class SequenceDataLoader(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(
        self,
        list_IDs,
        labels,
        image_dir,
        mask_dir,
        to_fit=True,
        batch_size=32,
        dim=(256, 256),
        n_channels=1,
        n_classes=10,
        shuffle=True,
    ):
        """Initialization

        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_dir: directory path to images location
        :param mask_dir: directory path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image_sequence = [
                self.image_dir + "/" + ID + "/{}.tif".format(year)
                for year in range(2016, 2023)
            ]
            X[i,] = self._load_grayscale_sequence(image_sequence)

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks

        :param list_IDs_temp: list of label ids to load
        :return: batch of masks
        """
        y = np.empty((self.batch_size, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            mask_sequence = [
                self.mask_dir + "/" + ID + "/{}.tif".format(year)
                for year in range(2016, 2023)
            ]
            y[i,] = self._load_grayscale_sequence(mask_sequence)

        return y

    def _load_grayscale_sequence(self, image_sequence):
        """Load grayscale image sequence and quantize to 9 colors

        :param image_sequence: list of paths to images to load as a sequence
        :return: loaded image sequence
        """
        images = [
            Image.open(img_path).convert("P", palette=Image.ADAPTIVE, colors=9)
            for img_path in image_sequence
        ]
        grayscale_images = [
            img.convert("L") for img in images
        ]  # Convert back to grayscale
        image_data = [
            np.array(img) / 255.0 for img in grayscale_images
        ]  # Convert to float and normalize
        return np.stack(image_data, axis=-1)

    def _load_grayscale_image(self, image_path):
        """Load grayscale image and quantize to 9 colors

        :param image_path: path to image to load
        :return: loaded image
        """
        img = Image.open(image_path).convert("P", palette=Image.ADAPTIVE, colors=9)
        img = img.convert("L")  # Convert back to grayscale
        img = np.array(img) / 255.0
        return img
