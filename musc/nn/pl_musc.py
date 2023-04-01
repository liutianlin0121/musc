""" PyTorch Lightning wrapper for MUSC model. """
from typing import List, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from musc.nn.sparse_coder import MUSC
# from lion_pytorch import Lion


class LitMUSC(pl.LightningModule):
    """PyTorch Lightning wrapper for MUSC model."""
    def __init__(  # pylint:disable=dangerous-default-value
        self,
        t_max: int,
        kernel_size: Optional[int] = 3,
        hidden_layer_widths: Optional[List[int]] = [512, 256, 128, 64, 32],
        n_classes: Optional[int] = 1,
        ista_num_steps: Optional[int] = 5,
        lasso_lambda_scalar: Optional[float] = 1e-3,
        tied_adjoint_encoder_bool: Optional[bool] = False,
        tied_encoder_decoder_bool: Optional[bool] = False,
        fixed_lambda_bool: Optional[bool] = False,
        relu_out_bool: Optional[bool] = False,
        nonneg_code_bool: Optional[bool] = True,
        eta_min: Optional[float] = 1e-6,
        weight_decay: Optional[float] = 0.0,
        learning_rate: Optional[float] = 2e-4,
        train_loss_fun=nn.MSELoss(),
        eval_loss_fun=nn.MSELoss(),
        eps=1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        model_setting_dict = {
            'kernel_size': kernel_size,
            'hidden_layer_widths': hidden_layer_widths,
            'n_classes': n_classes,
            'ista_num_steps': ista_num_steps,
            'lasso_lambda_scalar': lasso_lambda_scalar,
            'tied_adjoint_encoder_bool': tied_adjoint_encoder_bool,
            'tied_encoder_decoder_bool': tied_encoder_decoder_bool,
            'nonneg_code_bool': nonneg_code_bool,
            'fixed_lambda_bool': fixed_lambda_bool,
            'relu_out_bool': relu_out_bool}

        self.model = MUSC(**model_setting_dict)
        self.t_max = t_max
        self.t_warmup = t_max // 20
        self.eta_min = eta_min
        self.learning_rate = learning_rate
        self.train_loss_fun = train_loss_fun
        self.eval_loss_fun = eval_loss_fun
        self.eps = eps
        self.weight_decay = weight_decay

    def forward(self, x):
        # pylint:disable=missing-function-docstring
        # pylint:disable=invalid-name
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # pylint:disable=missing-function-docstring
        # pylint:disable=invalid-name
        # pylint:disable=unused-argument
        x, y, *_ = batch
        forward_output = self.model(x)
        loss = self.train_loss_fun(forward_output, y)
        self.log('hp_metric/train_loss',
                 loss,
                 prog_bar=True,
                 sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # pylint:disable=missing-function-docstring
        # pylint:disable=invalid-name
        x, y, *_ = batch
        output = self(x)
        val_metric = self.eval_loss_fun(output, y).detach()
        self.log("hp_metric/val_metric",
                 val_metric,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)

    def configure_optimizers(self):
        # pylint: disable=missing-function-docstring
        # pylint:disable=invalid-name
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            eps=self.eps)

        # optimizer = Lion(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     betas=(0.95, 0.98),
        #     weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.t_warmup,
            T_mult=(self.t_max - self.t_warmup)//self.t_warmup,
            eta_min=self.eta_min)
        return [optimizer], [scheduler]
