import xarray as xr
import dask
import numpy as np
import gcsfs
from typing import OrderedDict
import yaml
import time


def load_vars(var_config_file):
    with open(var_config_file, "r") as f:
        var_config = yaml.load(f, Loader=yaml.Loader)

    return var_config


def main():
    dir = 'era5_global_025degree'

    # load raw
    # ar_full_37_1h = xr.open_zarr(
    #     'gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr/'
    # )
    ar_full_37_1h = xr.open_zarr(
        'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/'
    )
    print("Model surface dataset size {} TiB".format(ar_full_37_1h.nbytes/(1024**4)))
    print(ar_full_37_1h)
    
    # static
    variables = OrderedDict(load_vars('./era5.yaml'))
    static = ar_full_37_1h[variables['input']['static']]
    print(static)
    fs = gcsfs.GCSFileSystem(project='colorful-aia')
    gcsmap = fs.get_mapper(f"weather-us-central1/{dir}/static.zarr")
    static.to_zarr(store=gcsmap)
    
    surface_data = ar_full_37_1h[variables['input']['surface']]
    high_data = ar_full_37_1h[variables['input']['high']].sel(level=variables['input']['levels'])
    
    for year in range(2021, 2022):
        time0 = time.time()
        print(year)
        #slice time
        sur = surface_data.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        hig = high_data.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        
        # merge variables
        sur_dat = xr.concat([sur['2m_temperature'],
                             sur['mean_sea_level_pressure'],
                             sur['10m_u_component_of_wind'],
                             sur['10m_v_component_of_wind'],
                             sur['total_precipitation']],
                             dim='variable')
        sur_dat = sur_dat.assign_coords(variable=variables['input']['surface'])

        hig_dat = xr.concat([hig['geopotential'],
                             hig['temperature'],
                             hig['specific_humidity'],
                             hig['u_component_of_wind'],
                             hig['v_component_of_wind']],
                             dim='variable')
        hig_dat = hig_dat.assign_coords(variable=variables['input']['high'])

        #transpose dims
        res_sur  = sur_dat.transpose('time','variable','latitude','longitude')
        c_surface = res_sur.to_dataset()
        c_surface = c_surface.rename({'2m_temperature': 'data'})
        
        res_high = hig_dat.transpose('time','variable','level','latitude','longitude')
        c_high = res_high.to_dataset()
        c_high = c_high.rename({'geopotential': 'data'})

        # store to GCS bucket
        fs = gcsfs.GCSFileSystem(project='colorful-aia')
        gcsmap = fs.get_mapper(f"weather-us-central1/{dir}/surface_{year}.zarr")
        c_surface.to_zarr(store=gcsmap)
        print(f"save surf data cost {time.time() - time0}")
        
        gcsmap = fs.get_mapper(f"weather-us-central1/{dir}/high_{year}.zarr")
        c_high.to_zarr(store=gcsmap)
        print(f"save high data cost {time.time() - time0}")
    print('OK')


if __name__ == '__main__':
    main()
