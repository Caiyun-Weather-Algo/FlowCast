import os
import numpy as np
import pandas as pd

_ERA5_DATA_ROOT = os.environ.get("ERA5_DATA_ROOT", "data/ERA5_GLOBAL")

def level_weights():
    pressure_weights = [1.4, 1.3, 1.2, 1.2, 1.2, 1.2, 1.0, 1.0, 1.0, 0.8, 0.8, 0.2, 0.1] * 5 
    surf_weights = [1.5] * 6
    weights = surf_weights + pressure_weights
    df = pd.read_csv(f"{_ERA5_DATA_ROOT}/var_weights.csv", index_col=0)
    var_weights = df['var_weights'].values
    weights = weights * var_weights
    return weights


def level_weights6():
    pressure_weights = [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 0.8, 0.8, 0.8, 0.8, 0.4, 0.2, 0.2] * 5 
    surf_weights = [1.5] * 6
    weights = surf_weights + pressure_weights
    df = pd.read_csv(f"{_ERA5_DATA_ROOT}/var_weights_6hr.csv", index_col=0)
    var_weights = df['var_weights'].values
    weights = weights * var_weights
    return weights


def level_weights6Pangu():  
    weights = [3.0, 1, 1.5, 0.77, 0.66, 1] + [2.8] * 13 + [1.7] * 13 + [0.78] * 13 + [0.87] * 13 + [0.6] * 13
    return weights


def _assert_increasing(x: np.ndarray):
  if not (np.diff(x) > 0).all():
    raise ValueError(f"array is not increasing: {x}")


def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
  pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
  return np.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])


def _cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
  """Calculate the area overlap as a function of latitude."""
  bounds = _latitude_cell_bounds(points)
  _assert_increasing(bounds)
  upper = bounds[1:]
  lower = bounds[:-1]
  # normalized cell area: integral from lower to upper of cos(latitude)
  return np.sin(upper) - np.sin(lower)


def _get_lat_weights(latitude):
  """Computes latitude/area weights from latitude coordinate of dataset."""
  weights = _cell_area_from_latitude(np.deg2rad(latitude))
  weights /= np.mean(weights)
  return weights

def latitude_weights(region="china"):
    if region == "china":
        latitudes = np.arange(60, 0 - 0.1, -0.25)
    elif region == "global":
        latitudes = np.arange(-90, 90 + 0.1, 1)
    elif region == "global025":
        latitudes = np.arange(90, -90 - 0.1, -0.25)
    else:
        latitudes = 0
    # cos_lat = np.cos(np.deg2rad(latitudes))
    # # Normalize latitude weights so they sum to 1
    # lat_weights = cos_lat / cos_lat.mean()
    lat_weights = _get_lat_weights(latitudes)
    return lat_weights


def weighted_acc(pred, y, mean, lat_weights):
    """
    Compute latitude-weighted ACC forvariable.
    
    :param pred: Predicted tensor of shape [1, H, W]
    :param y: Ground truth tensor of shape [1, H, W]
    :param latitudes: A 1D tensor or array of latitudes (in degrees) of shape [H]    
    """
    if len(pred.shape) ==2:
        pred = np.expand_dims(pred, 0)
    if len(y.shape) ==2:
        y = np.expand_dims(y, 0)
        
    y = y - mean # anomalies
    pred = pred - mean
    x = np.sum(y * pred * lat_weights[None, :, None])
    y = np.sqrt(np.sum(y * y * lat_weights[None, :, None]) * np.sum(pred * pred * lat_weights[None, :, None]))
    acc = x / y
    return acc


def weighted_rmse(pred, y, lat_weights):
    """
    Compute latitude-weighted RMSE forvariable.
    
    :param pred: Predicted tensor of shape [1, H, W]
    :param y: Ground truth tensor of shape [1, H, W]
    :param lat_weights: A 1D tensor or array of latitudes (in degrees) of shape [H]    
    """
    if len(pred.shape) ==2:
        pred = np.expand_dims(pred, 0)
    if len(y.shape) ==2:
        y = np.expand_dims(y, 0)
    
    num_lat = np.shape(pred)[1]
    num_lon = np.shape(pred)[2]
     
    x = np.sum((y-pred) ** 2 * lat_weights[None, :, None])
    rmse = np.sqrt(x / num_lat / num_lon)
    return rmse


if __name__ == '__main__':
    print(level_weights())