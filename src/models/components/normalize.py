import torch
import numpy as np
import xarray as xr
from einops import rearrange
from typing import Dict, List, Optional, Union


class DataNormalizer:
    def __init__(
        self,
        data_folder: str,
        config: Optional[Dict] = None,
        years: str = "2000_2019"
    ):
        """Initialize the DataNormalizer.
        
        Args:
            data_folder: Path to folder containing normalization statistics
            config: Optional custom config dict
            years: Year range for normalization statistics
        """
        self.data_folder = data_folder
        self.years = years
        
        # Default data configuration
        self.default_config = {
            'input': {
                'static': [
                    'geopotential_at_surface',
                    'land_sea_mask',
                    'soil_type'
                ],
                'surface': [
                    '2m_temperature',
                    'mean_sea_level_pressure',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind', 
                    'total_precipitation'
                ],
                'high': [
                    'geopotential',
                    'temperature',
                    'specific_humidity',
                    'u_component_of_wind',
                    'v_component_of_wind'
                ],
                'levels': [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
            }
        }
        
        # Use custom config if provided, otherwise use default
        self.input_vars = config['input'] if config else self.default_config['input']
        
        # Load normalization statistics
        self._load_statistics()
    
    def _load_statistics(self):
        """Load normalization statistics from files."""
        # Load difference statistics
        diff_std = xr.open_dataset(
            f"{self.data_folder}/diff_avgmax_{self.years}.nc",
            engine='netcdf4'
        )
        self.diff_std = diff_std.sel(level=self.input_vars["levels"])
        
        # Load raw statistics
        raw_mean = xr.open_dataset(
            f"{self.data_folder}/raw_mean_{self.years}.nc",
            engine='netcdf4'
        )
        self.raw_mean = raw_mean.sel(level=self.input_vars["levels"])
        
        raw_std = xr.open_dataset(
            f"{self.data_folder}/raw_std_{self.years}.nc",
            engine='netcdf4'
        )
        self.raw_std = raw_std.sel(level=self.input_vars["levels"])

    def normalize_data(
        self,
        data: Union[torch.Tensor, np.ndarray],
        type: str = "",
        reverse: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """Normalize or denormalize data.
        
        Args:
            data: Input data to normalize/denormalize
            type: Data type ('surface' or 'high')
            reverse: If True, denormalize instead of normalize
            
        Returns:
            Normalized or denormalized data
        """
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
            raise KeyError(f"Invalid type: {type}. Must be 'surface' or 'high'")
            
        return data

    def normalize(
        self,
        x: Union[torch.Tensor, np.ndarray],
        reverse: bool = False,
        data_pack: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Normalize full dataset.
        
        Args:
            x: Input data of shape (B, C, H, W)
            reverse: If True, denormalize instead of normalize
            data_pack: If True, pack surface and high data together
            
        Returns:
            Normalized data, either packed or as separate surface/high tensors
        """
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
            
    def normalize_diff(
        self,
        x: Union[torch.Tensor, np.ndarray],
        reverse: bool = False,
        data_pack: bool = True,
        scale_factor: float = 3.0
    ) -> Union[torch.Tensor, np.ndarray]:
        """Normalize differences between timesteps.
        
        Args:
            x: Input differences of shape (B, C, H, W)
            reverse: If True, denormalize instead of normalize
            data_pack: If True, pack surface and high data together
            scale_factor: Factor to scale the standard deviation
            
        Returns:
            Normalized differences, either packed or as separate surface/high tensors
        """
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
            std = self.diff_std[param_name].values / scale_factor
            if isinstance(x, torch.Tensor):
                std = torch.tensor(std).to(x.device)
            if not reverse:
                surf[:, i] /= std
            else:
                surf[:, i] *= std 
        
        for j in range(n_high_vars): 
            param_name = self.input_vars["high"][j]
            for l in range(n_levels):
                std = (self.diff_std[param_name].values[l]) / scale_factor
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