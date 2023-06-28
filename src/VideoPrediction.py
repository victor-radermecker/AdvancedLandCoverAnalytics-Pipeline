import numpy as np
from copy import deepcopy
from numpy.core.fromnumeric import take
from tqdm import tqdm

class VideoPrediction:

  def __init__(self, fishnet, filtered = False):

    self.fishnet_object = fishnet
    if filtered:
      self.fishnet = deepcopy(fishnet.filtered_fishnet)
    else:
      self.fishnet = deepcopy(fishnet.fishnet)

  def define_years_range(self, yearStart, yearEnd):

    self.years_range = range(yearStart, yearEnd + 1)

  def create_tensor(self):

    # keep track of first and last index
    first_flag_index = None
    last_flag_index = None
    condition_met_flag = False
    self.tensor = np.zeros((self.fishnet_object.fishnet_rows, self.fishnet_object.fishnet_cols, len(self.years_range)))
    for t, yr in enumerate(self.years_range):
      for i in tqdm(range(self.fishnet_object.fishnet_rows)):
        for j in range(self.fishnet_object.fishnet_cols):
          idx = self.fishnet_object.fishnet_cols * i + j
          if idx not in self.fishnet.index:
            continue
          self.tensor[i,j,t] = self.fishnet.loc[idx, f"MeanPixel_{yr}"]
          if not condition_met_flag:
            self.first_element = (i,j,t)
            condition_met_flag = True
          self.last_element = (i,j,t)

  def tensor_subset(self):

    # only take values computed by create_tensor
    self.tensor = self.tensor[
      self.first_element[0]:self.last_element[0]+1,
      self.first_element[1]:self.last_element[1]+1,
      self.first_element[2]:self.last_element[2]+1,
    ]


  """
  # Doesn't make sense to do this, much better to create the whole tensor (rows, cols, years) and then take slices on the go
  def create_tensors(self):
    position_name = {
            (-1,-1): 'UL', (-1,0): 'U', (-1,1): 'UR', 
            (0,-1): 'L', (0,1): 'R', 
            (1,-1): 'DL', (1,0): 'D', (1,1): 'DR',
        } 
    self.tensor = {}
    for _, row in tqdm(self.fishnet.iterrows()):
      tensor_row = np.zeros((3, 3, len(self.years_range)))
      for t, yr in enumerate(self.years_range):
        for i in range(0,3):
          for j in range(0,3):
            if i == 1 and j == 1:
              self.tensor[i,j,t] = row[f"MeanPixel_{yr}"]
            else:
              neighbor_id = row.neighbors[position_name[(i - 1, j - 1)]]
              neighbor_value = self.fishnet[self.fishnet["id"] == neighbor_id][f"MeanPixel_{yr}"]
              tensor_row[i,j,t] = neighbor_value.item() if len(neighbor_value) != 0 else 0
      self.tensor[row['id']] = tensor_row
  """
