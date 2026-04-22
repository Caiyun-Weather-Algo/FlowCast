
import numpy as np 
import xarray as xr
from datetime import datetime, timedelta
import time 
import dask
import os 
import gcsfs


def save_results_as_zarr(surf_array, high_array, timestamp, output_dir=""):
    mlat = 181
    nlon = 360
    t0 = time.time()
    # Example for initializing dimensions and coordinates
    variables = {'surface': ['2m_temperature', '2m_dewpoint_temperature', 'mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', 'total_precipitation'],
                 'high': ['geopotential', 'temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind'],
                 'levels': [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
                }
   
    # coords
    time_dim = np.array([datetime.fromtimestamp(timestamp)])
    prediction_timedelta_dim = np.array([timedelta(hours=i + 1) for i in range(surf_array.shape[0])])  # bug: hours should be i + 1
    latitude_dim = np.linspace(-90.0, 90.0, mlat)
    longitude_dim = np.linspace(0.0, 359.0, nlon)
    level_dim = variables['levels']

    # variables
    data_vars = {}
    # surf
    dims = ["time", "prediction_timedelta", "latitude", "longitude"]
    for i, var_name in enumerate(variables['surface']):
        var_data = surf_array[None, :, i]  # Extract the samples for the variable
        data_vars[var_name] = xr.DataArray(var_data,
                                           dims=dims,
                                           coords={
                                               "time": time_dim,
                                               "prediction_timedelta": prediction_timedelta_dim,
                                               "latitude": latitude_dim,
                                               "longitude": longitude_dim,
                                               })
    # high
    dims = ["time", "prediction_timedelta", "level", "latitude", "longitude"]
    for i, var_name in enumerate(variables['high']):
        var_data = high_array[None, :, i]  # Extract the samples for the variable
        data_vars[var_name] = xr.DataArray(var_data,
                                           dims=dims,
                                           coords={
                                               "time": time_dim,
                                               "prediction_timedelta": prediction_timedelta_dim,
                                               "latitude": latitude_dim,
                                               "longitude": longitude_dim,
                                               "level": level_dim
                                               })

    dataset = xr.Dataset(data_vars)
    timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H')
    zarr_name = f"flow_{timestamp_str}_hourly_lead_120.zarr"
    gcs_url = os.environ.get("GCS_ZARR_URL")  # e.g. "gs://bucket/prefix" — optional
    if gcs_url:
        fs = gcsfs.GCSFileSystem(
            project=os.environ.get("GCS_PROJECT", os.environ.get("GCP_PROJECT", "")) or None
        )
        prefix = gcs_url.rstrip("/")
        out = fs.get_mapper(f"{prefix}/{zarr_name}")
    else:
        out = os.path.join(output_dir, zarr_name)
    dataset.to_zarr(store=out, consolidated=True)
    print(f"Results saved to {zarr_name} in {time.time() - t0:.1f}s")