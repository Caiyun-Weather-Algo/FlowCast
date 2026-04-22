import numpy as np
import pandas as pd
import arrow
from einops import rearrange
import torch
from torch.utils.data import Dataset
import xarray as xr
import os
from src.utils.data_utils import get_latlon, get_local_time_features, tp2dbz
import time 
import dask


class ERA5Dataset(Dataset):
    """
    A PyTorch Dataset class to load ERA5 data for weather prediction models. 
    Loads input and output variables from ERA5 dataset based on specified 
    configuration files and time ranges. The dataset is designed for 
    multi-timestep input and forecasting.

    Args:
        start_time: str, the starting time for the dataset (format: YYYY-MM-DD_HH)
        end_time: str, the ending time for the dataset (format: YYYY-MM-DD_HH)
        input_vars: dict, dictionary of input variables (surface and pressure levels)
        output_vars: dict, dictionary of output variables (surface and pressure levels)
        mode: str, mode for dataset ('train', 'valid', 'test')
        input_step: int, number of historical timesteps used as input
        forecast_step: int, number of timesteps to forecast
        other optional arguments control data normalization, static variables, cloud data, etc.
    """

    def __init__(self,
                 root,
                 region,
                 resolution,
                 start_time,
                 end_time,
                 input_vars,
                 output_vars,
                 mode="train",
                 input_step=1,
                 start_lead=0,
                 forecast_step=6,
                 autoregressive_step=1,
                 sample_interval=1,
                 is_norm=False,
                 norm_method="minmax",
                 use_static=False,
                 add_latlon_time=False,
                ):
        self.root = root
        self.region = region
        self.resolution = resolution
        self.input_vars = input_vars
        self.output_vars = output_vars
        
        diff_std = xr.open_dataset(f"{self.data_folder}/diff_avgmax_1980_2019.nc", engine='netcdf4')
        self.diff_std = diff_std.sel(level=self.input_vars["levels"])
        raw_mean = xr.open_dataset(f"{self.data_folder}/raw_mean_1980_2019.nc", engine='netcdf4')
        self.raw_mean = raw_mean.sel(level=self.input_vars["levels"])
        raw_std = xr.open_dataset(f"{self.data_folder}/raw_std_1980_2019.nc", engine='netcdf4')
        self.raw_std = raw_std.sel(level=self.input_vars["levels"])

        self.forecast_step = forecast_step
        self.input_step = input_step
        self.start_lead = start_lead
        self.autoregressive_step = autoregressive_step
        self.mode = mode
        self.is_norm = is_norm
        self.norm_method = norm_method
        self.use_static = use_static
        self.add_latlon_time = add_latlon_time

        # Generate sample start times based on mode and interval
        self.sample_interval = sample_interval
        self.sample_start_t = self.gen_sample_times(start_time, end_time)
        self.year_start = int(start_time[0:4])
        self.year_end = int(end_time[0:4])
        self.year_start_stamp = int(arrow.get(f"{self.year_start}", "YYYY").timestamp())

        self.param_sfc = ["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation", "2m_dewpoint_temperature"]
        self.param_pl = ["geopotential", "temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]
        self.levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        surf_idxs = [self.param_sfc.index(k) for k in input_vars["surface"]]
        high_idxs = [self.param_pl.index(k) for k in input_vars["high"]]
        level_idxs = [self.levels.index(j) for j in self.input_vars["levels"]]
        self.input_var_idxs = {"surface": surf_idxs, "high": high_idxs, "level": level_idxs}

        # Load ERA5 data
        self.surf_data, self.dewpoint_temp_data, self.high_data, self.static_data = self._load_data()
        
        # static and coordinates (latitude/longitude) for optional additional features
        self.static = self._get_static()
        self.latlon = get_latlon(region=self.region, res=self.resolution)
        
        self.set_autoregressive_step()

    def _get_static(self, norm=True):
        static = []
        for var in self.input_vars['static']:
            static.append(self.static_data[var].to_numpy())
        static = np.array(static).astype(np.float32)  
        static = rearrange(static, 'v w h -> v h w', w=static.shape[1])
        
        for i in range(3):
            param_name = self.input_vars["static"][i]  
            mean = self.raw_mean[param_name].values.astype(np.float32)
            std = self.raw_std[param_name].values.astype(np.float32)
            if norm:
                static[i] = (static[i] - mean) / std
            else:
                static[i] = static[i] * std + mean
        return static
        
    def gen_sample_times(self, start_time, end_time, abandon_abnormal_dates=True):
        start_time_stamp = int(arrow.get(start_time, "YYYY-MM-DD HH:00").timestamp())
        end_time_stamp = int(arrow.get(end_time, "YYYY-MM-DD HH:00").timestamp())
        sample_start_t = list(range(start_time_stamp, end_time_stamp + 1, 3600))
        if self.mode != "train":
            sample_start_t = sample_start_t[::self.sample_interval]

        # # Filter out 09UTC and 21UTC samples
        if self.mode == "train":
            sample_start_t = [
                t for t in sample_start_t 
                # if arrow.get(t).format('HH') in ['00', '06', '12', '18']
                if arrow.get(t).format('HH') not in ['09', '21']
            ]

        if abandon_abnormal_dates:
            all_times = pd.read_csv(f'{self.data_folder}/abandon_date.csv', index_col=0).values[:, 0]
            sample_start_t = set(sample_start_t)
            exclude_timestamps = {
                int(arrow.get(t, "YYYY-MM-DD HH:00:00").timestamp())
                for t in all_times
            }
            sample_start_t = sample_start_t - exclude_timestamps
            sample_start_t = sorted(list(sample_start_t))
        return sample_start_t

    def _load_data(self):
        surf = [xr.open_dataset(f"gcs://weather-us-central1-v2/share/data/era5_global_1degree/surface_{yr}.zarr",
                                backend_kwargs={"storage_options": {"project": "colorful-aia", "token": None}},
                                engine="zarr", ) for yr in range(self.year_start, self.year_end + 1)]
        dewpoint_temp = [xr.open_dataset(f"gcs://weather-us-central1-v2/share/data/era5_global_1degree/2m_dewpoint_t_{yr}.zarr",
                                backend_kwargs={"storage_options": {"project": "colorful-aia", "token": None}},
                                engine="zarr", ) for yr in range(self.year_start, self.year_end + 1)]
        high = [xr.open_dataset(f"gcs://weather-us-central1-v2/share/data/era5_global_1degree/high_{yr}.zarr",
                                backend_kwargs={"storage_options": {"project": "colorful-aia", "token": None}},
                                engine="zarr", ) for yr in range(self.year_start, self.year_end + 1)]
        static = xr.open_dataset(f"gcs://weather-us-central1-v2/share/data/era5_global_1degree/static.zarr",
                                backend_kwargs={"storage_options": {"project": "colorful-aia", "token": None}},
                                engine="zarr")
        return surf, dewpoint_temp, high, static

    def get_time_idx(self, timestamp):
        yr = arrow.get(int(timestamp)).format("YYYY")
        yr_idx = int(eval(yr)-self.year_start)
        idx = int((timestamp - (arrow.get(yr, "YYYY").timestamp()))//3600)
        return yr_idx, idx

    def get_data(self, sample_time, left, right):
        surf = []
        high = []
        for idx in range(-1 * left, right):
            sample_start_time = sample_time + idx * 3600
            yr_idx, t_idx = self.get_time_idx(sample_start_time)
            high_data = self.high_data[yr_idx]['data'][t_idx].to_numpy()
            surf_data = self.surf_data[yr_idx]['data'][t_idx].to_numpy()
            dewpoint_temp_data = self.dewpoint_temp_data[yr_idx]['2m_dewpoint_temperature'][t_idx].to_numpy()
            surf_data = np.concatenate((surf_data, dewpoint_temp_data[None, :, :]), axis=0) # add dewpoint temperature to surface data
            surf_data = surf_data[self.input_var_idxs["surface"]]
            surf.append(surf_data)
            high.append(high_data)
            
        surf = np.array(surf)
        high = np.array(high)
        
        # sp
        high[:, 2] = np.where(high[:, 2] < 0, 0, high[:, 2])
        # tp
        surf[:, -1] = tp2dbz(surf[:, -1] * 1000)

        if self.is_norm:
            surf = self.normalize_data(surf, type="surface", reverse=False)
            high = self.normalize_data(high, type="high", reverse=False)
            
        # rearrange
        surf = rearrange(surf, 't v h w -> (t v) h w')
        high = rearrange(high, 't v d h w -> (t v d) h w')
        data = np.concatenate((surf, high), axis=0).astype(np.float32)
        
        # Convert to CPU tensor if needed
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        return data

    def set_autoregressive_step(self):
        """This is called once per batch to set a consistent autoregressive_step."""
        if self.mode == "train":
            self.current_batch_autoregressive_step = self.autoregressive_step # np.random.randint(1, self.autoregressive_step + 1)
        else:
            self.current_batch_autoregressive_step = self.autoregressive_step  # Fixed for non-training modes

    def __len__(self):
        return len(self.sample_start_t) - self.input_step + 1 - self.start_lead - self.forecast_step + 1

    def __getitem__(self, idx):
        """
        Get a sample from the dataset. Returns the input data (x_surf and x_high) and the target data.
        """
        self.set_autoregressive_step()
        idx = idx + self.input_step - 1
        autoregressive_step = self.current_batch_autoregressive_step
        sample_start_time = self.sample_start_t[idx]
        input = self.get_data(sample_start_time, self.input_step - 1, 1)
        
        output = []
        if self.mode != "predict":
            if self.mode != "train":
                iter_start = 0 
            else:
                iter_start = 0  # max(autoregressive_step - 2, 0)
            for iter in range(iter_start, autoregressive_step):
                target_start_time = sample_start_time + self.start_lead * 3600 + (self.start_lead + self.forecast_step - 1) * iter * 3600   
                y = self.get_data(target_start_time, 0, self.forecast_step)
                output.append(y)
            output = np.array(output).astype(np.float32)
        
        static = np.concatenate((self.static, self.latlon), axis=0).astype(np.float32)
        
        time_feature_list = []
        for iter in range(autoregressive_step):
            sample_time = sample_start_time + self.start_lead * 3600 + (self.start_lead + self.forecast_step - 1)  * iter * 3600
            time_features = get_local_time_features(arrow.get(int(sample_time)).format('YYYY-MM-DD HH:00'),
                                                    np.arange(0, 360, 1),
                                                    input.shape[1])
            time_feature_list.append(time_features)
        time_features = np.array(time_feature_list).astype(np.float32)
        
        return sample_start_time, input, output, static, time_features, autoregressive_step
    
    def normalize_data(self, data, type="", reverse=False):
        if type == 'surface':
            for i in range(len(self.input_vars["surface"])):
                param_name = self.input_vars["surface"][i]  
                mean = self.raw_mean[param_name].values
                std = self.raw_std[param_name].values
                if isinstance(data, torch.Tensor):
                    mean = torch.tensor(mean).to(data.device)
                    std = torch.tensor(std).to(data.device)
                if not reverse:
                    data[:, i] = (data[:, i] - mean) / std
                else:
                    data[:, i] = data[:, i] * std + mean
        elif type == "high":
            for j in range(len(self.input_vars["high"])): 
                param_name = self.input_vars["high"][j]
                for l in range(len(self.input_vars["levels"])):
                    mean = self.raw_mean[param_name].values[l]
                    std = self.raw_std[param_name].values[l]
                    if isinstance(data, torch.Tensor):
                        mean = torch.tensor(mean).to(data.device)
                        std = torch.tensor(std).to(data.device)
                    if not reverse:
                        data[:, j, l] = (data[:, j, l] - mean) / std
                    else:
                        data[:, j, l] = data[:, j, l] * std + mean  
        else:
            raise KeyError  
        return data
    
    def normalize(self, x, reverse=False, data_pack=True):
        "x (B, C, H, W)"
        n_surf_vars = len(self.input_vars["surface"])
        n_high_vars = len(self.input_vars["high"])
        n_levels = len(self.input_vars["levels"])
        
        if isinstance(x, torch.Tensor):
            surf = x[:, 0:n_surf_vars].clone()
            high = x[:, n_surf_vars:].clone()
        elif isinstance(x, np.ndarray):
            surf = x[:, 0:n_surf_vars].copy()
            high = x[:, n_surf_vars:].copy()
        else:
            raise ValueError(f"Invalid input type: {type(x)}")
        
        surf = rearrange(surf, 'b v h w -> b v h w')
        high = rearrange(high, 'b (v d) h w -> b v d h w', v=n_high_vars, d=n_levels)
  
        surf = self.normalize_data(surf, type="surface", reverse=reverse)
        high = self.normalize_data(high, type="high", reverse=reverse)
 
        if data_pack:
            surf = rearrange(surf, 'b v h w -> b v h w')
            high = rearrange(high, 'b v d h w -> b (v d) h w')
            if isinstance(x, torch.Tensor):
                return torch.cat((surf, high), dim=1)
            else:
                return np.concatenate((surf, high), axis=1)
        else:
            return surf.astype(np.float32), high.astype(np.float32)
            
    def normalize_diff(self, x, reverse=False, data_pack=True):
        "x (B, C, H, W)"
        n_surf_vars = len(self.input_vars["surface"])
        n_high_vars = len(self.input_vars["high"])
        n_levels = len(self.input_vars["levels"])
        
        if isinstance(x, torch.Tensor):
            surf = x[:, 0:n_surf_vars].clone()
            high = x[:, n_surf_vars:].clone()
        elif isinstance(x, np.ndarray):
            surf = x[:, 0:n_surf_vars].copy()
            high = x[:, n_surf_vars:].copy()
        else:
            raise ValueError(f"Invalid input type: {type(x)}")
        
        surf = rearrange(surf, 'b v h w -> b v h w')
        high = rearrange(high, 'b (v d) h w -> b v d h w', v=n_high_vars, d=n_levels)
        
        for i in range(n_surf_vars):
            param_name = self.input_vars["surface"][i]     
            std = self.diff_std[param_name].values / 3
            if isinstance(x, torch.Tensor):
                std = torch.tensor(std).to(x.device)
            if not reverse:
                surf[:, i] /= std
            else:
                surf[:, i] *= std 
        
        for j in range(n_high_vars): 
            param_name = self.input_vars["high"][j]
            for l in range(n_levels):
                std = (self.diff_std[param_name].values[l]) / 3
                if isinstance(x, torch.Tensor):
                    std = torch.tensor(std).to(x.device)
                if not reverse:
                    high[:, j, l] /= std
                else:
                    high[:, j, l] *= std
        
        if data_pack:
            surf = rearrange(surf, 'b v h w -> b v h w')
            high = rearrange(high, 'b v d h w -> b (v d) h w')
            if isinstance(x, torch.Tensor):
                return torch.cat((surf, high), dim=1)
            else:
                return np.concatenate((surf, high), axis=1)
        else:
            return surf.astype(np.float32), high.astype(np.float32)
        
    @property
    def data_folder(self) -> str:
        return os.path.join(self.root, "ERA5_GLOBAL")
