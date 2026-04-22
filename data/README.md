Place preprocessed ERA5 (or your dataset) under this directory and point `data_dir` and datamodule options in the Hydra configs to the right subpaths. This repository does not ship large NetCDF or Zarr; obtain data under the license of the product you use, and cite it in your work.

`ERA5_DATA_ROOT` (default `data/ERA5_GLOBAL`) is read by some utilities, including **variable weight CSVs** used in `src/utils/metrics.py` (e.g. `var_weights*.csv` matching your stage). Add those small tables next to your stats files when you run training.
