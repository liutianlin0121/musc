""" Train the CT inverse problem model. """
import argparse
from torch import nn
from torch.nn.functional import mse_loss
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from musc.dataloaders.load_lodopab_ct import get_dataloaders_ct
from musc.nn.pl_musc import LitMUSC
from musc.utils import get_musc_root


parser = argparse.ArgumentParser(description='MUSC training')

# Dataset settings
parser.add_argument('--BATCH_SIZE', type=int, default=2,
                    help='mini-batch size for training.')
parser.add_argument('--IMAGE_CHANNEL_NUM', type=int, default=1,
                    help='number of channels of an image.')
parser.add_argument('--TRAIN_PERCENT', type=int, default=100,
                    help='number of channels of an image.')

# Model settings
parser.add_argument('--task_name', type=str, default='ct',
                    help='the name of the model to be saved.')
parser.add_argument('--KERNEL_SIZE', type=int, default=3,
                    help='the kernel size used for 2D convolution.')
parser.add_argument('--HIDDEN_WIDTHS', type=int, nargs='+',
                    default=[512, 256, 128, 64, 32], help='')
parser.add_argument('--ISTA_NUM_STEPS', type=int, default=5,
                    help='number of ISTA iterations.')
parser.add_argument('--LASSO_LAMBDA_SCALAR_LIST',type=float,
                    default=[0.0, 0.0, 0.0, 0.01, 0.1],
                    help='initialized LASSO parameter.')
parser.add_argument('--RELU_OUT_BOOL', default=True,
                    type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--TIED_ADJOINT_ENCODER_BOOL', default=False,
                    type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--TIED_ENCODER_DECODER_BOOL', default=False,
                    type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--FIXED_LAMBDA_BOOL', default=False,
                    type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--NONNEG_CODE_BOOL', default=True,
                    type=lambda x: (str(x).lower() == 'true'))

# Training settings
parser.add_argument('--NUM_EPOCH', type=int, default=50,
                    help='number of training epochs.')
parser.add_argument('--LOSS_STR', type=str, default='nn.MSELoss()',
                    help='the loss function from the torch.nn module.')
parser.add_argument('--LEARNING_RATE', type=float, default=2e-4,
                    help='learning rate of gradient descent.')
parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-6)
parser.add_argument('--GRADIENT_CLIP_VAL', type=float, default=1e-2,
                    help='gradient clipping value')
parser.add_argument('--ETA_MIN', type=float, default=1e-5,
                    help='the eta_min for CosineAnnealingLR decay.')
parser.add_argument('--EPS', type=float, default=5e-8)
parser.add_argument('--MIXED_PRECISION', default=False,
                    type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use mixed precision training')


# DistributedDataParallel settings
parser.add_argument('--num_workers', type=int, default=20,
                    help='')
parser.add_argument('--gpu_devices', type=int, nargs='+', default=[0, 1],
                    help='')


def main():
  """Main function."""
  pl.seed_everything(1234)
  args = parser.parse_args()

  dataset_setting_dict = {
    'batch_size': args.BATCH_SIZE,
    'num_workers': args.num_workers,
    'train_percent': args.TRAIN_PERCENT}

  loaders = get_dataloaders_ct(**dataset_setting_dict)

  if args.LOSS_STR == 'nn.MSELoss()':
    train_loss_fun = nn.MSELoss()
  elif args.LOSS_STR == 'nn.L1Loss()':
    train_loss_fun = nn.L1Loss()
  else:
    raise ValueError('Invalid loss function.')

  def my_psnr(x_input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Function that computes PSNR
    """
    mse_val = mse_loss(x_input, target, reduction='mean')
    max_val_tensor: torch.Tensor = torch.max(
        target).clone().detach().to(x_input.device).to(x_input.dtype)
    return 10 * torch.log10(max_val_tensor * max_val_tensor / mse_val)

  model_setting_dict = {
      'kernel_size': args.KERNEL_SIZE,
      'hidden_layer_widths': args.HIDDEN_WIDTHS,
      'n_classes': args.IMAGE_CHANNEL_NUM,
      'ista_num_steps': args.ISTA_NUM_STEPS,
      'lasso_lambda_scalar_list': args.LASSO_LAMBDA_SCALAR_LIST,
      'tied_adjoint_encoder_bool': args.TIED_ADJOINT_ENCODER_BOOL,
      'tied_encoder_decoder_bool': args.TIED_ENCODER_DECODER_BOOL,
      'nonneg_code_bool': args.NONNEG_CODE_BOOL,
      'fixed_lambda_bool': args.FIXED_LAMBDA_BOOL,
      'learning_rate': args.LEARNING_RATE,
      'train_loss_fun': train_loss_fun,
      'weight_decay': args.WEIGHT_DECAY,
      'eval_loss_fun': my_psnr,
      'relu_out_bool': args.RELU_OUT_BOOL,
      'eta_min': args.ETA_MIN,
      't_max': args.NUM_EPOCH * len(loaders['train']) // len(args.gpu_devices),
      'eps': args.EPS
      }
  print(model_setting_dict)
  model = LitMUSC(**model_setting_dict)
  checkpoint_callback = ModelCheckpoint(
      monitor='val_metric',
      save_top_k=1,
      mode='max')

  if args.MIXED_PRECISION:
    precision = 16
  else:
    precision = 32

  model_save_dir = str(get_musc_root()) + '/saved_models/'
  logger = TensorBoardLogger(
      model_save_dir,
      name=args.task_name,
      )
  logger.log_hyperparams(model.hparams)

  lr_monitor = LearningRateMonitor(logging_interval='step')
  trainer = pl.Trainer(
    profiler='simple',
    logger=logger,
    # track_grad_norm=2,
    devices=args.gpu_devices,
    accelerator='gpu',
    strategy='ddp',
    # strategy='deepspeed_stage_2',
    #'ddp_sharded', #'ddp', #DDPStrategy(find_unused_parameters=False),
    max_epochs=args.NUM_EPOCH,
    gradient_clip_val=args.GRADIENT_CLIP_VAL,
    callbacks=[checkpoint_callback, lr_monitor],
    default_root_dir=model_save_dir + '/' + args.task_name,
    precision=precision
    )

  trainer.fit(model,
              train_dataloaders=loaders['train'],
              val_dataloaders=loaders['validation'])


if __name__ == '__main__':
  main()
