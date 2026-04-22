import numpy as np
import pandas as pd


def load_static(data_folder, var="z"):
    file = f"{data_folder}/china_{var}.npz"
    data = np.load(file)[var]
    return data


def get_coords(region="china", res=0.25, degree=True):
    if region == "china":
        lonmin, lonmax = 70, 140
        latmin, latmax = 0, 60
        res = 0.25
    if region == "global":
        lonmin, lonmax = 0, 359
        latmin, latmax = -90, 90
        
    lats = np.arange(latmin, latmax + res, res)[::-1]
    lons = np.arange(lonmin, lonmax + res, res)
    coords = np.array(np.meshgrid(lons, lats))

    if degree:
        return coords
    else:
        return np.deg2rad(coords)


def get_latlon(region="china", res=0.25):
    if region == "china":
        lonmin, lonmax = 70, 140
        latmin, latmax = 0, 60

    if region == "global":
        lonmin, lonmax = 0, 360
        latmin, latmax = -90, 90

    lats = np.linspace(latmin, latmax, int((latmax - latmin) / res + 1))
    lons = np.linspace(lonmin, lonmax - res, int((lonmax - lonmin) / res))
    lons_2d, lats_2d = np.array(np.meshgrid(lons, lats))

    sin_lat = np.sin(np.deg2rad(lats_2d))
    cos_lat = np.cos(np.deg2rad(lats_2d))
    sin_lon = np.sin(np.deg2rad(lons_2d))
    cos_lon = np.cos(np.deg2rad(lons_2d))
    return np.array([sin_lat, cos_lat, sin_lon, cos_lon])
    # return np.array([sin_lat, sin_lon, cos_lon])
   
    
def get_time_features(t):
    cur_time = pd.to_datetime(t) # 'YYYY-MM-DD HH:00'
    y_h = [cur_time.day_of_year / 366 * 2 * np.pi, cur_time.hour / 24 * 2 * np.pi]
    y_h = np.array(y_h, dtype=np.float32)
    y_h = np.concatenate([np.sin(y_h), np.cos(y_h)], axis=-1)
    time_features = y_h.reshape(-1)
    return time_features


def get_local_time_features(t, longitude, lat_dim):
    lon_dim = longitude.shape[0]
    cur_time = pd.to_datetime(t)  # 'YYYY-MM-DD HH:00'
    year_progress = np.array(cur_time.day_of_year / 366)
    day_progress_greenwich = np.array(cur_time.hour / 24)

    # Offset the day progress to the longitude of each point on Earth.
    longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
    day_progress = np.mod(
        day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0
    )

    year_features = featurelize(year_progress)
    day_features = featurelize(day_progress)
    
    year_features = np.broadcast_to(year_features[:, np.newaxis, np.newaxis],
                                                (year_features.shape[0], lat_dim, lon_dim))
    day_features = np.broadcast_to(day_features[:, np.newaxis, :],
                                                (day_features.shape[0], lat_dim, lon_dim))
    return np.concatenate([year_features, day_features], axis=0)


def featurelize(progress):
    progress = progress * (2 * np.pi)
    return np.array([np.sin(progress), np.cos(progress)])
    
    
def tp2dbz(x, dmin=2, dmax=55):
    # x[x < 0.04] = 0.04
    x[x < 0.04] = 0
    y = 10 * np.log10(1 + 200 * np.power(x, 1.6))
    # y[y < dmin] = dmin    # 2 dbz ~ 0.05 mm/h
    # y[y > dmax] = dmax    # 55 dbz ~ 100 mm/h
    return y


def log_transform(x, alpha=1e-3, reverse=False):
    # x: m/h
    if not reverse:
        x[x < 0] = 0
        x = np.log10(1 + x / alpha)
    else:
        x = (np.power(10, x) - 1) * alpha
    return x
