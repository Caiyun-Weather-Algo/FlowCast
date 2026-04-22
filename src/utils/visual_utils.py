import matplotlib.pyplot as plt
import numpy as np 
import torch 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors  
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
from cartopy.util import add_cyclic_point


def define_map(region="era5"):    
    if region == "global":
        lonmin, lonmax = 0, 359
        latmin, latmax = -90, 90
        extents = [lonmin, lonmax, latmin, latmax]
        res = 1
    lats = np.arange(latmin, latmax + res, res)
    lons = np.arange(lonmin, lonmax + res, res)
    return extents, lats, lons 


def plot_raw_and_incre(x0, y, samples_x0, samples, var=None, levels=None, filename=""):
    s, m, n = x0.shape
    fig, axes = plt.subplots(s, 3, figsize=(15, int(s*3)), constrained_layout=True)

    # generate grid
    XX, YY = np.meshgrid(np.arange(n + 1), np.arange(m + 1))
    cmap = plt.get_cmap('viridis', 25)
    
    for i in range(s):
        if levels:
            title = f"{var}_{levels[i]}hPa"
        else:
            title = f"{var[i]}"
    
        data_min = min(x0[i].min(), y[i].min(), samples[i].min())
        data_max = max(x0[i].max(), y[i].max(), samples[i].max()) 
        vmin = data_min
        vmax = data_max
        
        im = axes[i][0].pcolormesh(XX, YY, x0[i], cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
        axes[i][0].set_title(f'X0 - {title}')

        axes[i][1].pcolormesh(XX, YY, y[i], cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
        axes[i][1].set_title(f'X1 - {title}')
        
        axes[i][2].pcolormesh(XX, YY, samples[i], cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
        axes[i][2].set_title(f'Sampled - {title}')
    
        fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.02, pad=0.04)
    plt.savefig(f"{filename}.png", dpi=300)
    plt.close(fig)
    
    # plot increment
    fig, axes = plt.subplots(s, 2, figsize=(10, int(s*3)), constrained_layout=True)
    cmap = plt.get_cmap('RdBu_r', 25)

    for i in range(s):
        if levels:
            title = f"{var}_{levels[i]}hPa"
        else:
            title = f"{var[i]}"
    
        data_min = min((y[i] - x0[i]).min(), (samples[i] - samples_x0[i]).min())
        data_max = max((y[i] - x0[i]).max(), (samples[i] - samples_x0[i]).max())
        vmin = data_min
        vmax = data_max
        im = axes[i][0].pcolormesh(XX, YY, y[i] - x0[i], cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
        axes[i][0].set_title(f'target - {title}')
        
        axes[i][1].pcolormesh(XX, YY, samples[i] - samples_x0[i], cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
        axes[i][1].set_title(f'predict - {title}')
    
        fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.02, pad=0.04)
    plt.savefig(f"{filename}_incre.png", dpi=300)
    plt.close(fig)
    

def plot_error(x0, y1, y2, var=None, levels=None, filename=""):
    _, m, n = x0.shape

    # generate grid
    XX, YY = np.meshgrid(np.arange(n + 1), np.arange(m + 1))
   
    # plot increment
    fig, axes = plt.subplots(5, 2, figsize=(10, 15), constrained_layout=True)
    cmap = plt.get_cmap('RdBu_r', 25)

    for i in range(5):
        if levels:
            title = f"{var}_{levels[i]}hPa"
        else:
            title = f"{var[i]}"
    
        data_min = min((y1[i] - x0[i]).min(), (y2[i] - x0[i]).min())
        data_max = max((y1[i] - x0[i]).max(), (y2[i] - x0[i]).max())
        vmin = max(data_min, -1*data_max)
        vmax = min(data_max, -1*data_min)
        im = axes[i][0].pcolormesh(XX, YY, y1[i] - x0[i], cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
        axes[i][0].set_title(f'pangu error - {title}')
        
        axes[i][1].pcolormesh(XX, YY, y2[i] - x0[i], cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
        axes[i][1].set_title(f'flow error - {title}')
    
        fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.02, pad=0.04)
    plt.savefig(f"{filename}_error.png", dpi=600)
    plt.close(fig)
    
    
def plot_contour_map(ax, fig, lons, lats, data, title,  cmap=None, cbar=True):
    """
    Plot contour map
    """
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    
    cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
    
    im = ax.pcolormesh(
        cyclic_lons, lats, cyclic_data, 
        transform=ccrs.PlateCarree(),
        cmap=cmap, 
        alpha=1,
    )
    if cbar:
        cbar = fig.colorbar(
            im, ax=ax, shrink=0.4, pad=0.1, orientation='vertical',
        )
        cbar.ax.tick_params(labelsize='small')
   
    ax.set_title(title)
    return im 

    
def plot_global_withmap(x0, y, samples_x0, samples, region, var=None, levels=None, filename=""):
    _, lats, lons = define_map(region=region)
    cmap = plt.get_cmap('viridis', 30)
    proj = ccrs.Robinson(central_longitude=180)

    # fig, axes = plt.subplots(5, 3, figsize=(15, 15), subplot_kw={'projection': proj}, constrained_layout=True)
    for i in range(5):
        if levels:
            title = f"{var}_{levels[i]}hPa"
        else:
            title = f"{var[i]}"
        print(title)
        fig = plt.figure(figsize=(8, 10), constrained_layout=True)

        ax1 = fig.add_subplot(3, 1, 1, projection=proj)
        im1 = plot_contour_map(ax1, fig, lons, lats, x0[i], f'X0 - {title}', cmap, cbar=False)
        ax2 = fig.add_subplot(3, 1, 2, projection=proj)
        im2 = plot_contour_map(ax2, fig, lons, lats, y[i], f'X1 - {title}', cmap, cbar=False)
        ax3 = fig.add_subplot(3, 1, 3, projection=proj)
        im3 = plot_contour_map(ax3, fig, lons, lats, samples[i], f'Sampled - {title}', cmap, cbar=False)

        cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], orientation='vertical', pad=0.02, shrink=0.6)
        cbar.ax.tick_params(labelsize='small')
            
        plt.savefig(f"{filename}_{title}.png", dpi=300)
        plt.close(fig)
    
    # plot increment
    cmap = plt.get_cmap('RdBu_r', 30)
    for i in range(5):
        if levels:
            title = f"{var}_{levels[i]}hPa"
        else:
            title = f"{var[i]}"
        fig = plt.figure(figsize=(8, 10), constrained_layout=True)

        ax1 = fig.add_subplot(2, 1, 1, projection=proj)
        im1 = plot_contour_map(ax1, fig, lons, lats, y[i] - x0[i], f'target - {title}', cmap, cbar=False)
        ax2 = fig.add_subplot(2, 2, 2, projection=proj)
        im2 = plot_contour_map(ax2, fig, lons, lats, samples[i] - samples_x0[i], f'predict - {title}', cmap, cbar=False)

        cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', pad=0.02, shrink=0.6)
        cbar.ax.tick_params(labelsize='small')
    
        plt.savefig(f"{filename}_{title}_incre.png", dpi=300)
        plt.close(fig)