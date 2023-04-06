"""Utility functions for nn modules."""
from torch import nn


def get_all_conv(net, conv_list):
  """Get all nn.Conv2d layers in a network."""
  for _, layer in net._modules.items():   # pylint: disable=protected-access
    if not isinstance(layer, nn.Conv2d):
      get_all_conv(layer, conv_list)
    elif isinstance(layer, nn.Conv2d):
      # it's a Conv layer. Register a hook
      conv_list.append(layer)
  return conv_list


def get_all_transpose_conv(net, conv_list):
  """Get all nn.ConvTranspose2d layers in a network."""
  for _, layer in net._modules.items():    # pylint: disable=protected-access
    if not isinstance(layer, nn.ConvTranspose2d):
      get_all_transpose_conv(layer, conv_list)
    elif isinstance(layer, nn.ConvTranspose2d):
      # it's a ConvTranspose2d layer. Register a hook
      conv_list.append(layer)
  return conv_list


def get_all_conv_and_transpose_conv(net):
  """Get all nn.Conv2d and nn.ConvTranspose2d layers in a network."""
  conv_list = get_all_conv(net, conv_list=[])
  transposed_conv_list = get_all_transpose_conv(net, conv_list=[])
  return conv_list + transposed_conv_list
