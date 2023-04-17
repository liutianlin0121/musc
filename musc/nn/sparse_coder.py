"""Sparse coding module"""
from typing import List, Optional
import torch.nn as nn
import torch
from torch.nn.functional import softplus
from musc.nn.dictionary import UDictionary, TransposedUDictionary
from musc.nn.activations import relu, softshrink
from musc.nn.utils import get_all_conv_and_transpose_conv

INITIALIZER = torch.nn.init.kaiming_uniform_


class MUSC(nn.Module):
  """Multiscale U-Net like sparse coding

  musc_model = MUSC(
        kernel_size = 3,
        hidden_layer_widths = [1024, 512, 256, 128, 64],
        n_classes = 1,
        ista_num_steps = 5,
        lasso_lambda_scalar = 1e-3
        )
  random_out = musc_model(torch.rand(1, 1, 128, 128))
  """

  def __init__(self,
               kernel_size: int,
               hidden_layer_widths: List[int],
               n_classes: int,
               ista_num_steps: int,
               lasso_lambda_scalar_list: List[float],
               tied_adjoint_encoder_bool: Optional[bool] = False,
               tied_encoder_decoder_bool: Optional[bool] = False,
               nonneg_code_bool: Optional[bool] = True,
               fixed_lambda_bool: Optional[bool] = False,
               relu_out_bool: Optional[bool] = False):

    super().__init__()

    self.n_classes = n_classes
    self.ista_num_steps = ista_num_steps
    self.lasso_lambda_scalar_list = lasso_lambda_scalar_list
    self.hidden_layer_widths = hidden_layer_widths
    assert len(self.lasso_lambda_scalar_list) == len(self.hidden_layer_widths)

    self.num_scales = len(hidden_layer_widths)
    self.encoder_dictionary = UDictionary(
      kernel_size, hidden_layer_widths,
      n_classes)

    if nonneg_code_bool:
      self.thres_operator = relu
    else:
      self.thres_operator = softshrink

    for conv in get_all_conv_and_transpose_conv(self.encoder_dictionary):
      INITIALIZER(conv.weight, mode='fan_in', nonlinearity='linear')

    # Configure the encoder dictionary
    if tied_adjoint_encoder_bool:
      self.precond_encoder_dictionary = self.encoder_dictionary
    else:
      self.precond_encoder_dictionary = UDictionary(
          kernel_size, hidden_layer_widths, n_classes)
      # initialize with the same atoms
      self.precond_encoder_dictionary.load_state_dict(
        self.encoder_dictionary.state_dict())

    self.adjoint_precond_encoder_dictionary = TransposedUDictionary(
        self.precond_encoder_dictionary)

    # Configure the decoder dictionary
    if tied_encoder_decoder_bool:
      self.decoder_dictionary = self.encoder_dictionary

    else:
      self.decoder_dictionary = UDictionary(
        kernel_size,
        hidden_layer_widths,
        n_classes)
      # initialize with the same atoms
      self.decoder_dictionary.load_state_dict(
        self.encoder_dictionary.state_dict())

    if relu_out_bool:
      self.out_transform = nn.ReLU()
    else:
      self.out_transform = nn.Identity()

    # a list of lambdas, one for each iteration
    lasso_lambda_iter_list = [[
        torch.nn.Parameter(
            torch.full((1, hidden_layer_widths[i], 1, 1),
                       self.lasso_lambda_scalar_list[i]))
        for i, _ in enumerate(hidden_layer_widths)
    ] for _ in range(ista_num_steps)]

    # a list of lambdas, with length equal to:
    # (the number of scales) x (the number of iterations)
    self.lasso_lambda_iter_list = nn.ParameterList(
        [item for sublist in lasso_lambda_iter_list for item in sublist])

    if fixed_lambda_bool:
      for param in self.lasso_lambda_iter_list.parameters():
        param.requires_grad = False

    # self.ista_stepsizes = torch.nn.ParameterList(
    #     [nn.Parameter(torch.ones(1) * 0.6) for i in range(self.ista_num_steps)])

    # a list of stepsizes, one for each iteration
    ista_stepsizes_iter_list = [[
        torch.nn.Parameter(
            torch.ones((1, hidden_layer_widths[i], 1, 1)) * 0.6 )
        for i, _ in enumerate(hidden_layer_widths)
    ] for _ in range(ista_num_steps)]

    # a list of stepsizes, with length equal to:
    # (the number of scales) x (the number of iterations)
    self.ista_stepsizes_iter_list = nn.ParameterList(
        [item for sublist in ista_stepsizes_iter_list for item in sublist])


  def forward(self, x):
    # pylint: disable=missing-function-docstring
    # pylint:disable=invalid-name
    alphas = self.run_ista_steps(x)
    output = self.decoder_dictionary(alphas)
    output = self.out_transform(output)
    return output

  def run_ista_steps(self, z):
    """ISTA steps
    alpha_{k+1} = S_{lambda}(alpha_k + stepsize * (D^T z - D^T D alpha_{k}))
    """
    # pylint:disable=invalid-name
    D = self.encoder_dictionary
    Dt = self.adjoint_precond_encoder_dictionary  # D^{\top}
    Dtz = Dt(z)  # D^{\top}z

    for step in range(self.ista_num_steps):
      # make sure that the stepsize is positive
      if step == 0:
        # \alpha <= stepsize * D^{\top}z
        alphas =  []
        for scale in range(self.num_scales):
          stepsize = softplus(self.ista_stepsizes_iter_list[scale])
          alphas.append(stepsize * Dtz[scale])
      else:
        # \alpha <= \alpha + stepsize * (D^{\top}z - D^{\top}D \alpha))
        DtDalphas = Dt(D(alphas))
        for scale in range(self.num_scales):
          stepsize = softplus(
            self.ista_stepsizes_iter_list[step * self.num_scales + scale])
          alphas[scale] = \
            alphas[scale] + stepsize * (Dtz[scale] - DtDalphas[scale])

      for scale in range(self.num_scales):
        stepsize = softplus(
          self.ista_stepsizes_iter_list[step * self.num_scales + scale])
        alphas[scale] = self.thres_operator(
            alphas[scale],
            lambd=stepsize * self.lasso_lambda_iter_list[
              step * self.num_scales + scale])
    return alphas
