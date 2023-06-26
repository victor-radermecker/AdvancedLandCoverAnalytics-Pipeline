import numpy as np
from copy import deepcopy

class XGB:

  def __init__(self, fishnet, filtered = False):
    """
    Add docstrings, fishnet is fishnet_complete
    """
    self.fishnet = fishnet
    if filtered:
      self.df = deepcopy(fishnet.filtered_fishnet)
    else:
      self.df = deepcopy(fishnet.fishnet)
    
  def feature_engineering(self, yearStart, yearEnd):
    self.years_range = range(yearStart, yearEnd + 1)
    for yr in self.years_range[:-1]:
      for feature in ["MeanPixel", "Entropy"]:
        if feature == "Entropy" and yr == self.years_range[-1]:
            continue
        self.df[f"Δ{feature}_{yr}_{yr+1}"] = self.df[f"{feature}_{yr}"] - self.df[f"{feature}_{yr+1}"]

  def remove_original_features(self):
    to_drop = [f"{feature}_{yr}" for yr in self.years_range for feature in ["MeanPixel", "Entropy"]]
    self.df.drop(columns = to_drop, axis = 1, inplace = True)


    #def plot_results(self):

    