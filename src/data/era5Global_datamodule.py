from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, SubsetRandomSampler
from src.data.components.era5_global import ERA5Dataset


class ERA5DataModule(LightningDataModule):
    """
    LightningDataModule` for the ERA5 dataset.
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        pin_memory,
        dataset_split,
        dataset_cfg,
    ) -> None:
        """Initialize a `ERA5`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.create_dataset(start_time=self.hparams.dataset_split.train[0],
                                                  end_time=self.hparams.dataset_split.train[1],
                                                  sample_interval=self.hparams.dataset_split.train[2],
                                                  mode="train"
                                                  )
            self.data_val = self.create_dataset(start_time=self.hparams.dataset_split.val[0],
                                                end_time=self.hparams.dataset_split.val[1],
                                                sample_interval=self.hparams.dataset_split.val[2],
                                                mode="validate"
                                                )
            self.data_test = self.create_dataset(start_time=self.hparams.dataset_split.test[0],
                                                 end_time=self.hparams.dataset_split.test[1],
                                                 sample_interval=self.hparams.dataset_split.test[2],
                                                 mode="test"
                                                )
            self.data_predict = self.create_dataset(start_time=self.hparams.dataset_split.predict[0],
                                                 end_time=self.hparams.dataset_split.predict[1],
                                                 sample_interval=self.hparams.dataset_split.predict[2],
                                                 mode="predict"
                                                 )

    def create_dataset(self, start_time, end_time, sample_interval, mode="train"):
        dataset_cfg = self.hparams.dataset_cfg
        return ERA5Dataset(root=self.hparams.data_dir,
                           region=dataset_cfg.region,
                           resolution=dataset_cfg.resolution,
                           start_time=start_time,
                           end_time=end_time,
                           mode=mode,
                           input_vars=dataset_cfg.input,
                           output_vars=dataset_cfg.input,
                           input_step=dataset_cfg.input_step,
                           start_lead=dataset_cfg.start_lead,
                           forecast_step=dataset_cfg.forecast_step,
                           autoregressive_step=dataset_cfg.autoregressive_step,
                           sample_interval=sample_interval,
                           is_norm= dataset_cfg.norm,
                           norm_method=dataset_cfg.norm_method,
                           use_static=dataset_cfg.use_static,
                           add_latlon_time=dataset_cfg.add_latlon_time,
                         )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        sample_interval = self.hparams.dataset_split.train[2]
        if sample_interval > 1:
            indices = np.random.choice(len(self.data_train), len(self.data_train)//sample_interval, replace=False)
            sampler = SubsetRandomSampler(indices)
            shuffle = False
        else: 
            sampler = None
            shuffle = True
        # else:
        #     if self.trainer.world_size  > 0: 
        #         sampler = torch.utils.data.distributed.DistributedSampler(
        #                 self.data_train,
        #                 shuffle=False,
        #                 seed=4
        #             )
        #     else:
        #         sampler = None
            
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0,
        )
    
    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0,
        )


if __name__ == "__main__":
    _ = ERA5DataModule()
