import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.utils import Sequence
import os
import imageio


class SequenceDataLoader(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(
        self,
        labels,
        list_IDs,  # regionIDs
        target,
        tile_region_dic,  # each key: batchID and each value is a list of fishnet IDs
        fishnet_coordinates,  # each key: fishnetID, each value: image coordinates within the region of each fishnet
        image_dir,
        dim,
        batch_size=32,
        n_channels=1,
        shuffle=True,
    ):
        """Initialization

        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_dir: directory path to images location
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.labels = labels
        self.list_IDs = list_IDs  # name of all batch_IDs
        self.target = target
        self.tile_region_dic = tile_region_dic
        self.fishnet_coordinates = fishnet_coordinates
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self._init_params()
        self.on_epoch_end()

    def _init_params(self):
        self.N_regions = len(self.tile_region_dic)
        self.N_fishnets = len(self.fishnet_coordinates)
        self.N_labels = len(self.labels)

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

        # Find list of IDs --> These are the batch_IDs that we want to load in this batch
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)
        y = self._generate_y(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_y(self, list_IDs_temp):
        # Initialization
        regionLengths = [len(self.tile_region_dic[ID]) for ID in list_IDs_temp]
        y = np.empty(sum(regionLengths))

        # Generate data
        cursor = 0
        for regionID in list_IDs_temp:
            for fishnetID in self.tile_region_dic[regionID]:
                y[cursor] = self.target[fishnetID]
                cursor += 1

        return y

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        regionLengths = [len(self.tile_region_dic[ID]) for ID in list_IDs_temp]
        X = np.empty((sum(regionLengths), len(self.labels), *self.dim, self.n_channels))

        # Generate data
        cursor = 0
        for i, ID in enumerate(list_IDs_temp):
            X[cursor : cursor + regionLengths[i], :] = self._load_region(ID)
            cursor += regionLengths[i]

        return X

    def _load_region(self, regionID):
        """
        Load a region from the image directory
        """
        # N is the number of fishnet in that region
        fishnetIDs = self.tile_region_dic[regionID]
        N = len(fishnetIDs)
        X = np.empty(
            (N, self.N_labels, self.dim[0], self.dim[1], self.n_channels)
        )  # dim [X, 6, 40, 44, 1]

        for i, label in enumerate(self.labels):
            region_path = os.path.join(
                self.image_dir,
                str(label),
                "Final",
                f"{regionID}.tif",
            )
            img = Image.open(region_path)
            if self.n_channels == 1:
                img = img.convert("L")  # convert to grayscale
            img = np.array(img) / 255.0

            for j, fishnetID in enumerate(fishnetIDs):
                coordinates = self.fishnet_coordinates[fishnetID]
                sub_img = self._crop_image(img, fishnetID, regionID, coordinates)
                X[j, i, :, :, :] = np.array(sub_img).reshape(
                    self.dim[0], self.dim[1], self.n_channels
                )

        return X

    def _crop_image(self, image, tile_id, batch_id, coordinates):
        xmin, ymin, xmax, ymax = coordinates
        subimage = image[ymin:ymax, xmin:xmax]

        if subimage.shape[0] < 30 or subimage.shape[1] < 30:
            raise Exception(
                f"Subimage {tile_id} in batch {batch_id} is too small. Please check the image and fishnet."
            )
        subimage = subimage[: self.dim[0], : self.dim[1]]
        return subimage

    ###################################################################################################

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
