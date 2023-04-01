"""Sparse coding module"""
from typing import List, Optional
import torch.nn as nn
import torch
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
                 lasso_lambda_scalar: float,
                 tied_adjoint_encoder_bool: Optional[bool] = False,
                 tied_encoder_decoder_bool: Optional[bool] = False,
                 nonneg_code_bool: Optional[bool] = True,
                 fixed_lambda_bool: Optional[bool] = False,
                 relu_out_bool: Optional[bool] = False
                 ):
        super().__init__()

        self.n_classes = n_classes
        self.ista_num_steps = ista_num_steps
        self.lasso_lambda_scalar = lasso_lambda_scalar
        self.hidden_layer_widths = hidden_layer_widths
        self.num_scales = len(hidden_layer_widths)
        self.encoder_dictionary = UDictionary(
            kernel_size, hidden_layer_widths, n_classes)

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
                kernel_size, hidden_layer_widths, n_classes)
            # initialize with the same atoms
            self.decoder_dictionary.load_state_dict(
                self.encoder_dictionary.state_dict())

        if relu_out_bool:
            self.out_transform = nn.ReLU()
        else:
            self.out_transform = nn.Identity()

        # a list of lambdas, one for each iteration
        _lasso_lambda_iter_list = [[torch.nn.Parameter(
            torch.full((1, width, 1, 1), lasso_lambda_scalar))
            for width in hidden_layer_widths] for _ in range(ista_num_steps)]

        # a list of lambdas, with length equal to:
        # (the number of scales) x (the number of iterations)
        self.lasso_lambda_iter_list = nn.ParameterList(
            [item for sublist in _lasso_lambda_iter_list for item in sublist])

        if fixed_lambda_bool:
            for param in self.lasso_lambda_iter_list.parameters():
                param.requires_grad = False

    def ista_steps(self, x):
        """ISTA steps
        alpha_{k+1} = S_{lambda}(alpha_k + D^T x - D^T D alpha_{k})
        """
        # pylint:disable=invalid-name
        D = self.encoder_dictionary
        Dt = self.adjoint_precond_encoder_dictionary  # D^{\top}
        Dt_x = Dt(x)  # D^{\top}x

        for step in range(self.ista_num_steps):
            if step == 0:
                # D^{\top}x
                code_list = [x for x in Dt_x]
            else:
                # \alpha_k + D^{\top}x - D^{\top}D \alpha_{k})
                code_list = [
                    code_list[scale] + Dt_x[scale] - Dt(D(code_list))[scale]
                    for scale in range(self.num_scales)
                    ]

            for scale in range(self.num_scales):
                code_list[scale] = self.thres_operator(
                    code_list[scale],
                    lambd=self.lasso_lambda_iter_list[
                        step * self.num_scales + scale])
        return code_list

    def forward(self, x):
        # pylint: disable=missing-function-docstring
        # pylint:disable=invalid-name
        code_list = self.ista_steps(x)
        output = self.decoder_dictionary(code_list)
        output = self.out_transform(output)
        return output
