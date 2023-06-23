import numpy as np

class XGB:

  def __init__(self, fishnet, filtered = False):
    """
    Add docstrings, fishnet is fishnet_complete
    """
    self.fishnet = fishnet
    if filtered:
      self.df = fishnet.filtered_fishnet
    else:
      self.df = fishnet.fishnet

################################
# CHECK AND SEE IF CAN IMPROVE #
################################
    
  def feature_engineering(self, yearStart, yearEnd):
    self.years_range = range(yearStart, yearEnd)
    for yr in self.years_range:
      for feature in ["MeanPixel", "Entropy"]:
        if feature == "Entropy":
          if yr == self.years_range[-1]:
            continue

        yr_feature = np.array(self.df[f"{feature}_{yr}"])
        next_yr_feature = np.array(self.df[f"{feature}_{yr+1}"])
        self.df[f"Delta1_{feature}_{yr}_{yr+1}"] = next_yr_feature - yr_feature
        self.df[f"Delta2_mean_{yr}_{yr+1}"] = (next_yr_feature - yr_feature) / (next_yr_feature + yr_feature)

  def remove_original_features(self):
    to_drop = []
    for yr in self.years_range:
      for feature in ["MeanPixel", "Entropy"]:
        if feature == "Entropy":
          if yr == self.years_range[-1]:
            continue
        to_drop.append(f"{feature}_{yr}")
    self.df.drop(columns = to_drop, axis = 1, inplace = True)


    #def plot_results(self):

    