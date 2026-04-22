from typing import Any, Dict, Tuple
from omegaconf import DictConfig, OmegaConf
import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from datetime import datetime
import os 
from math import ceil
import arrow
from torch_ema import ExponentialMovingAverage as EMA
import pandas as pd 
from einops import rearrange
from src.utils.metrics import latitude_weights, weighted_acc, weighted_rmse, level_weights6Swin3d
from src.utils.eval_util import cal_acc_rmse
from src.utils.visual_utils import plot_raw_and_incre
from src.utils.data_utils import log_transform
from src.utils.result_util import save_results_as_zarr
from src.utils.total_energy import compute_energy_components, compute_grid_area
from src.utils.spectrum import compute_spectrum
import numpy as np 


class FlowMatchingModule(LightningModule):
    """
    Flow Matching
    """

    def __init__(
        self,
        net: torch.nn.Module,
        flow,
        solver,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        training_cfg: DictConfig,
        inference_cfg: DictConfig,
        compile: bool,
        result_dir: str,
        automatic_optimization: bool = False
    ) -> None:
        """Initialize a rectified flow network.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
            
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_cfg = training_cfg
        self.inference_cfg = inference_cfg
        
        self.flow = flow
        self.solver = solver
        self.ema = EMA(self.net.parameters(), decay=self.training_cfg.ema_decay) if self.ema_wanted else None
        
        # Per-element MSE; combine with weights in model_step (reduction='none' avoids w^2 in loss).
        self.criterion = torch.nn.MSELoss(reduction="none")
        region = 'global'
        self.loss_weights = torch.tensor(latitude_weights(region), dtype=torch.float32)
        self.level_weights = torch.tensor(level_weights6Swin3d(), dtype=torch.float32) if self.training_cfg.is_level_weights else None

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

        # automatic optimization
        self.automatic_optimization = automatic_optimization     
        
    @property
    def ema_wanted(self):
        return self.training_cfg.ema_decay != -1
    
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if (self.trainer.testing or self.trainer.predicting) and self.ema_wanted:
        # if self.ema_wanted:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        return super().on_load_checkpoint(checkpoint)

    def on_before_zero_grad(self, optimizer) -> None:
        if self.ema_wanted:
            self.ema.update(self.net.parameters())
        return super().on_before_zero_grad(optimizer)

    def to(self, *args, **kwargs):
        if self.training_cfg.ema_decay != -1:
            self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x_t, t*1000, static)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.iter_per_epoch = ceil(len(self.trainer.datamodule.data_train) / (self.trainer.datamodule.hparams.batch_size))      
        print(self.iter_per_epoch)
        self.val_loss.reset()

    @staticmethod
    def sample_training_timestep(B, device, method="uniform"):
        if method == "uniform":
            t = torch.rand((B), device=device)
        elif method == "lognormal":
            m = 0
            s = 1
            u = torch.normal(m, s, size=(B,), device=device)
            t = torch.exp(u) / (1 + torch.exp(u))  
        elif method == "uniform_discrete":
            N = 1000
            t_values = torch.arange(0, 1, 1.0/N, device=device) 
            t_indices = torch.randint(0, len(t_values), (B,), device=device)  # Random indices
            t = t_values[t_indices]
        elif method == "uniform_discrete100":
            N = 100
            t_values = torch.arange(0, 1, 1.0/N, device=device) 
            t_indices = torch.randint(0, len(t_values), (B,), device=device)  # Random indices
            t = t_values[t_indices]
        else:
            t = None
        return t
    
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
        t = self.sample_training_timestep(x.shape[0], x.device, method=self.training_cfg.t_schedule)
        x0 = dataset.normalize(x)
        x1 = dataset.normalize(y[:, 0])
        x_t = self.flow.get_conditional_flow(x0, x1, t)
        target = self.flow.get_conditional_vector_field(x0, x1)
      
        # train network
        pred = self.forward(x_t, t, static)
        weights = self.weights.to(x.device)
        loss = (self.criterion(pred, target) * weights).mean()
        return loss, pred, y

    def model_finetune_multistep_manual(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """fine-tune model of multistep on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """  
        init_time, x0, y0, static0, time_features, total_autoregressive_step = batch
        dataset = self.trainer.datamodule.data_train
        N = 1
        autoregressive_step_period = 6
        # dt = 1. / N
        dt = 1. / (N * autoregressive_step_period)
        B, *x_shape = x0.size()

        # normalize
        x = dataset.normalize(x0)
        y0 = rearrange(y0, 'b a c h w -> a b c h w')
        y = torch.zeros_like(y0)
        for t in range(y.shape[0]):
            y[t] = dataset.normalize(y0[t])
        del x0, y0
        # forward
        total_loss = 0.0
        accumulate_grad_batches = 1
        scaling_factors = torch.pow(1 + torch.arange(1, 1+total_autoregressive_step[0], device=x.device) / 24, -0.5)

        for autoregressive_step in range(total_autoregressive_step[0]):
            # X_t-1 to X_t
            x = x.detach().requires_grad_()
            static_indice = int((autoregressive_step // 6) * 6 + 5)
            # static_indice = autoregressive_step 
            static = torch.cat([static0, time_features[:, static_indice]], dim=1)
            for i in range(N):
                # t = torch.ones((B), device=x.device) *  i / N 
                t = torch.ones((B), device=x.device) * (i + autoregressive_step % 6 * N) / (N * autoregressive_step_period)
                pred_step = self.forward(x, t, static)
                x = x + pred_step * dt
            y_pred = x
            
            # loss
            scaling_factor = scaling_factors[autoregressive_step]
            weights = self.weights.to(x.device) * scaling_factor # * hour_weights
            loss = (self.criterion(y_pred, y[autoregressive_step]) * weights).mean()
            
            # backward
            self.manual_backward(loss/total_autoregressive_step[0])
            total_loss += loss.item()/total_autoregressive_step[0]

        if (batch_idx + 1) % accumulate_grad_batches == 0:
            self.optimizers().step()
            if self.ema_wanted:
                self.ema.update(self.net.parameters()) 
            self.optimizers().zero_grad()

        return total_loss, y_pred, y
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        self.lr_schedulers().step(self.current_epoch + batch_idx / self.iter_per_epoch)
        # current_lr = self.optimizers().param_groups[0]['lr']
        # self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)

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
        if self.ema_wanted:
            with self.ema.average_parameters():
                loss, preds, targets = self.model_step(batch)
                # loss, preds, targets = self.model_step_linear(batch)
                # loss, _, _ = self.model_finetune_innerstep(batch, batch_idx, 'valid')
                # loss, _, _ = self.model_finetune_multistep(batch)

        else:
            loss, preds, targets = self.model_step(batch)
            # loss, preds, targets = self.model_step_linear(batch)
            # loss, _, _ = self.model_finetune_innerstep(batch, batch_idx, 'valid')
            # loss, _, _ = self.model_finetune_multistep(batch)
        
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
        output_dir = f"{self.hparams.result_dir}/test_lead24_new_ep{self.current_epoch}_{self.inference_cfg.sampler}_step{self.inference_cfg.num_sampling_step}"
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
            output_dir = f"{self.hparams.result_dir}/test_lead24_new_ep{self.current_epoch}_{self.inference_cfg.sampler}_step{self.inference_cfg.num_sampling_step}"
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
        optimizer = self.hparams.optimizer(params=self.net.parameters())
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
        solver = self.solver(self.net, self.ema)
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
                # Perform sampling with ODE method
                static = torch.cat([static0, time_features[:, iter]], dim=1)
                # static = torch.cat([static0, time_features[:, -1]], dim=1)
                # static = torch.cat([static0, time_features[:, (iter // 6) * 6 + 5]], dim=1)
                samples = solver.sampling(
                                        x0=x0,
                                        N=self.inference_cfg.num_sampling_step ,
                                        s=static,
                                        sampler=self.inference_cfg.sampler,
                                        dataset=dataset,
                                        autoregressive_step=iter % 6
                                        )
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
                    
                    # plot_flag = False
                    # if plot_flag and k == 0:
                    if iter % 24 == 0 and plot_flag and k == 0:
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
        solver = self.solver(self.net, self.ema)
        dataset = self.trainer.datamodule.data_predict

        timestamp, x0, _, static0, time_features, autoregressive_step = batch
        x0 = x0.to(self.device)
        static0 = static0.to(self.device)
        time_features = time_features.to(self.device)

        surf_samples, high_samples = [], []
        for iter in range(autoregressive_step[0]):
            #static = torch.cat([static0, time_features[:, iter]], dim=1)
            static = torch.cat([static0, time_features[:, (iter // 6) * 6 + 5]], dim=1)

            sample = solver.sampling(
                x0=x0,
                N=self.inference_cfg.num_sampling_step,
                s=static,
                sampler=self.inference_cfg.sampler,
                dataset=dataset,
                autoregressive_step=iter % 6
            )

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
    _ = FlowMatchingModule(None, None, None, None)
