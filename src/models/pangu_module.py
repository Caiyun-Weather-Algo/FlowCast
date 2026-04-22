from typing import Any, Dict, Tuple
from omegaconf import DictConfig, OmegaConf
import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from datetime import datetime
import os 
from math import ceil
from torch_ema import ExponentialMovingAverage as EMA
import pandas as pd 
from einops import rearrange
from src.utils.metrics import latitude_weights, weighted_acc, weighted_rmse, level_weights6Pangu
from src.utils.eval_util import cal_acc_rmse
from src.utils.visual_utils import plot_raw_and_incre
from src.utils.result_util import save_results_as_zarr
import numpy as np 


class PanguModule(LightningModule):
    """
    Pangu
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        result_dir: str,
    ) -> None:
        """Initialize a Pangu network.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
            
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # loss function
        self.criterion = torch.nn.MSELoss()
        region = 'global'
        self.loss_weights = torch.tensor(latitude_weights(region), dtype=torch.float32)
        self.level_weights = torch.tensor(level_weights6Pangu(), dtype=torch.float32)

        latitude_weights_clamped = torch.clamp(self.loss_weights, min=0.1)
        latitude_weights_clamped = latitude_weights_clamped/latitude_weights_clamped.mean()
        self.weights = latitude_weights_clamped[None, None, :, None] * self.level_weights[None, :, None,  None]
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.test_outputs = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.iter_per_epoch = ceil(len(self.trainer.datamodule.data_train) / (self.trainer.datamodule.hparams.batch_size))      
        print(self.iter_per_epoch)
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        _, x, y, static, time_features, autoregressive_step = batch
        dataset = self.trainer.datamodule.data_train 

        static = torch.cat([static, time_features[:, 0]], dim=1)
        x = dataset.normalize(x)
        y = dataset.normalize(y[:, 0])
        # target = y
        # pred difference
        target = y
        # train network
        pred = self.forward(x)
        weights = self.weights.to(x.device)
        loss = self.criterion(pred * weights, target * weights)
        return loss, pred, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        self.lr_schedulers().step(self.current_epoch + batch_idx / self.iter_per_epoch)
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, logger=True, on_step=False, on_epoch=True, prog_bar=True)

        # plot
        if batch_idx == 0 and self.current_epoch % 1 ==0:
            output_dir = f"{self.hparams.result_dir}/valid_ep{self.current_epoch}"
            os.makedirs(output_dir, exist_ok=True)  
            exp_name = self.hparams.result_dir.split("/exp_")[-1]          
            metrics_acc, metrics_rmse, keys = self.sample_and_plot(batch, every_n_within_batch=100, plot_flag=True, output_dir=output_dir)
            self.save_metrics(metrics_acc, metrics_rmse, keys, output_dir, f"{exp_name}_rank{self.trainer.local_rank}")
            
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), logger=True, sync_dist=True, prog_bar=True)
      
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        output_dir = f"{self.hparams.result_dir}/test_ep{self.current_epoch}"
        os.makedirs(output_dir, exist_ok=True)
        plot_flag = True if batch_idx==0 else False
        
        metrics_acc, metrics_rmse, keys = self.sample_and_plot(batch, every_n_within_batch=1, plot_flag=plot_flag, output_dir=output_dir)
        self.test_outputs.append({"acc": metrics_acc, "rmse": metrics_rmse, "keys": keys})
    
    def on_test_epoch_end(self) -> None:
        gathered_outputs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_outputs, self.test_outputs)
        test_outputs = [output for gpu_outputs in gathered_outputs for output in gpu_outputs]

        global_metrics_acc = {}
        global_metrics_rmse = {}
        for output in test_outputs:
            global_metrics_acc.update(output["acc"])
            global_metrics_rmse.update(output["rmse"])
        global_keys = output["keys"]   
         
        # Save results with average values
        if self.trainer.is_global_zero:
            output_dir = f"{self.hparams.result_dir}/test_ep{self.current_epoch}"
            os.makedirs(output_dir, exist_ok=True)
            exp_name = self.hparams.result_dir.split("/exp_")[-1]
            self.save_metrics(global_metrics_acc, global_metrics_rmse, global_keys, output_dir, exp_name)
        self.test_outputs.clear()
        
    def save_metrics(self, metrics_acc, metrics_rmse, keys, output_dir, exp_name):
        """Save accuracy and RMSE metrics to CSV with average values."""
        # Save accuracy metrics
        df_acc = pd.DataFrame(metrics_acc, index=keys).T
        df_acc.loc['average'] = df_acc.mean(axis=0)
        df_acc.sort_index().to_csv(f'{output_dir}/acc_{exp_name}.csv')

        # Save RMSE metrics
        df_rmse = pd.DataFrame(metrics_rmse, index=keys).T
        df_rmse.loc['average'] = df_rmse.mean(axis=0)
        df_rmse.sort_index().to_csv(f'{output_dir}/rmse_{exp_name}.csv')
        
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    @torch.no_grad()
    def sample_and_plot(self, batch, every_n_within_batch=24, plot_flag=False, output_dir=""):
        """Sample and plot results at the end of each validation epoch."""
        # Use a batch from validation set for sampling
        dataset = self.trainer.datamodule.data_train

        for member in range(1):
            metrics_acc = {}
            metrics_rmse = {}

            timestamp, x0, y, static0, time_features, autoregressive_step = batch
            x0 = x0.to(self.device)
            y = y.to(self.device)
            static0 = static0.to(self.device)
            time_features = time_features.to(self.device)
            
            forecast_step = dataset.forecast_step # currently, model's forecast step is 1 in training process
            y_surf = y[:, :, 0:6*forecast_step].cpu().numpy()
            y_high = y[:, :, 6*forecast_step:].cpu().numpy()
            y_surf = rearrange(y_surf, 'b a (t v) h w -> b a t v h w', a=autoregressive_step[0], t=forecast_step)
            y_high = rearrange(y_high, 'b a (t v d) h w -> b a t v d h w', a=autoregressive_step[0], t=forecast_step, v=5, d=13)
            
            for iter in range(autoregressive_step[0]):
                # X_t-1 to X_t
                static = torch.cat([static0, time_features[:, iter]], dim=1)
                x0_norm = dataset.normalize(x0, reverse=False, data_pack=True)
                pred = self.forward(x0_norm)
                samples = dataset.normalize(pred, reverse=True, data_pack=True) 
                # tp > 0
                samples[:, 5] = torch.where(samples[:, 5] < 3.3, 0, samples[:, 5])
                # q >  0
                samples[:, 32:45] = torch.where(samples[:, 32:45] < 0, 0, samples[:, 32:45])

                if iter == 0:
                    surf_x0, high_x0 = unpack(x0)
                else:
                    surf_x0, high_x0 = y_surf[:, iter-1, 0], y_high[:, iter-1, 0]
                surf_samples_x0, high_sampels_x0 = unpack(x0)  # x0 for iterative sampling    
                surf_y, high_y = y_surf[:, iter, 0], y_high[:, iter, 0] #unpack(y)
                surf_samples, high_samples = unpack(samples)
                                 
                x0 = samples

                for k in range(0, x0.shape[0], every_n_within_batch):
                    # time
                    timestamp_str = datetime.fromtimestamp(timestamp[k]).strftime('%Y-%m-%d_%H')
                    
                    if plot_flag and k == 0 and iter % 4==0:
                        # plot surface
                        plot_raw_and_incre(surf_x0[k], surf_y[k], surf_samples_x0[k], surf_samples[k], dataset.input_vars["surface"], None,
                                    filename = f"{output_dir}/{timestamp_str}_autoregressive{iter}_surface_ens{member}")
                    
                        # plot high
                        selected_levels = [925, 850, 700, 500, 200]
                        level_indices = [j for j, level in enumerate(dataset.input_vars["levels"]) if level in selected_levels]
                        for i, var in enumerate(dataset.input_vars["high"]):
                            plot_raw_and_incre(high_x0[k, i, level_indices], high_y[k, i, level_indices], high_sampels_x0[k, i, level_indices], high_samples[k, i, level_indices], var, levels=selected_levels, 
                                        filename=f"{output_dir}/{timestamp_str}_autoregressive{iter}_{var}_ens{member}")
                        
                    # cal rmse and acc
                    acc, rmse, keys = cal_acc_rmse(self.loss_weights.numpy(), surf_samples[k], surf_y[k], high_samples[k], high_y[k], dataset)
                    metrics_acc[f'{timestamp_str}_autoregressive{iter}'] = acc
                    metrics_rmse[f'{timestamp_str}_autoregressive{iter}'] = rmse                
        return metrics_acc, metrics_rmse, keys
    
    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        """precdition"""
        dataset = self.trainer.datamodule.data_predict

        timestamp, x0, _, static0, time_features, autoregressive_step = batch
        x0 = x0.to(self.device)
        static0 = static0.to(self.device)
        time_features = time_features.to(self.device)

        surf_samples, high_samples = [], []
        for iter in range(autoregressive_step[0]):
            static = torch.cat([static0, time_features[:, iter]], dim=1)
            x0_norm = dataset.normalize(x0, reverse=False, data_pack=True)
            pred = self.forward(x0_norm)
            sample = dataset.normalize(pred, reverse=True, data_pack=True) 
            # tp > 0
            sample[:, 5] = torch.where(sample[:, 5] < 3.3, 0, sample[:, 5])
            # q >  0
            sample[:, 32:45] = torch.where(sample[:, 32:45] < 0, 0, sample[:, 32:45])
                                
            x0 = sample
            surf_sample, high_sample = unpack(sample)
            surf_samples.append(surf_sample)
            high_samples.append(high_sample)

        surf_array = np.array(surf_samples)  # lead_time * b * nvar * mlat * nlon
        high_array = np.array(high_samples)  # lead_time * b * nvar * nlevel * mlat * nlon
    
        for k in range(0, x0.shape[0]):
            output_dir = f"{self.hparams.result_dir}/data"
            os.makedirs(output_dir, exist_ok=True)
            save_results_as_zarr(surf_array[:, k], high_array[:, k], timestamp[k], output_dir)
            
    
def unpack(x):
    surf = x[:, 0:6]
    high = x[:, 6:]
    surf = rearrange(surf, 'b v h w -> b v h w')
    high = rearrange(high, 'b (v d) h w -> b v d h w', v=5, d=13)
    return surf.cpu().numpy(), high.cpu().numpy()
    

if __name__ == "__main__":
    _ = SwinTransformer3DModule(None, None, None, None)
