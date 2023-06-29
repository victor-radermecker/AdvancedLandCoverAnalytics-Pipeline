import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.utils import indexable
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV


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
    self.seed = 42
    
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

  def neighbors_features(self):
    position_names = ['UL', 'U', 'UR', 'L', 'R', 'DL', 'D', 'DR']
    for yr in self.years_range[1:-1]:
      for pos_name in tqdm(position_names):
        self.df[f"{pos_name}_{yr}"] = self.df.apply(lambda row: self.get_neighbor_value(row, pos_name, yr), axis=1)

  def get_neighbor_value(self, row, pos_name, yr):
    if pos_name not in row["neighbors"]:
      return 0 # that neighbor doesn't exist
    else:
      neighbor_id = row["neighbors"][pos_name]
      values = self.df.loc[self.df["id"] == neighbor_id, f"ΔMeanPixel_{yr-1}_{yr}"].values
      return values[0] if len(values) != 0 else 0

  def train_model(self):
    target = f'ΔMeanPixel_{self.years_range[-1] - 1}_{self.years_range[-1]}'
    X = self.df.drop(columns = [target], axis = 1, inplace = False)
    y = self.df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = self.seed, train_size = 0.8)

    xgb_para = {'learning_rate': [0.1],
                'alpha': [0.1, 0.2, 0.3, 0.4],
                'max_depth': [4],
                'min_child_weight': [1,4],
                'gamma': [i/10.0 for i in range(0,5,2)],
            }
    self.model = GridSearchCV(
        xgb.XGBRegressor(random_state = self.seed), xgb_para, cv=5,
    )
    self.model.fit(X_train, y_train)
    self.make_predictions(X_train, y_train) # in-sample
    self.make_predictions(X_test, y_test) # out-of-sample

  def make_predictions(self, data, truth):
    pred = self.model.predict(data)
    rmse = np.sqrt(mean_squared_error(truth, pred))
    r2 = r2_score(truth, pred)
    plot_results(truth, pred, r2, rmse)

########################
### OTHER  FUNCTIONS ###
########################

def plot_results(y_test, y_pred, r2, rmse):
  # plot the predicted vs true values
  plt.scatter(y_test, y_pred)
  plt.xlabel("True Values")
  plt.ylabel("Predictions")
  plt.title(f"R^2 Score: {r2:.2f}, RMSE: {rmse:.2f}")
  plt.show()