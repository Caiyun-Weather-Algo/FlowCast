from src.utils.metrics import weighted_acc, weighted_rmse
import pandas as pd 


def cal_acc_rmse(weights, surf_pred, surf_y, high_pred, high_y, dataset): 
    surf_vars = dataset.input_vars["surface"]
    high_vars = dataset.input_vars["high"]
    levels = dataset.input_vars["levels"]
    stats = dataset.raw_mean #dataset.data_stats
    
    acc = []
    rmse = []
    keys = []
    
    level = "surface"
    for i, var in enumerate(surf_vars):
        pred, y = surf_pred[i], surf_y[i]
        mean = stats[var].values
        acc.append(weighted_acc(pred, y, mean, weights))
        rmse.append(weighted_rmse(pred, y, weights))
        keys.append(var)
        
    level = "high"
    for i, var in enumerate(high_vars):
        for j, level_idx in enumerate(levels):
            pred, y = high_pred[i, j], high_y[i, j]
            mean = stats[var].values[j]
            acc.append(weighted_acc(pred, y, mean, weights))
            rmse.append(weighted_rmse(pred, y, weights))
            keys.append(f"{var}_{levels[j]}")
    return acc, rmse, keys    