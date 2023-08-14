import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, base_path, dataframe, dtype=torch.float32, transform=None):
        self.base_path = base_path
        self.dataframe = dataframe
        self.dtype = dtype
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.get_image_path(index)
        image = Image.open(image_path).convert("RGB")
        image = torch.from_numpy(np.array(image))
        image = image.permute(2, 0, 1).type(self.dtype)

        if self.transform is not None:
            image = self.transform(image)
        target = self.get_target(index)
        return image, target

    def get_image_path(self, index):
        tile_id, year = self.get_tile_year(index)
        image_path = f"{self.base_path}/{year}/{tile_id}.tif"
        return image_path

    def get_target(self, index):
        tile_id, year = self.get_tile_year(index)
        if tile_id not in self.dataframe["tile_id"].values:
            raise ValueError(f"Tile ID {tile_id} not found in dataframe.")
        target = self.dataframe[
            (self.dataframe["tile_id"] == tile_id) & (self.dataframe["year"] == year)
        ][f"urbanization_rate"]
        return torch.Tensor(target.values.reshape(-1))

    def get_tile_year(self, index):
        tile_id = self.dataframe.iloc[index]["tile_id"]
        year = self.dataframe.iloc[index]["year"]
        return tile_id, year
