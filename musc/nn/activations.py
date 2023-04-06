"""Activation functions for PyTorch."""
import torch.nn as nn


def relu(x, lambd):   # pylint: disable=invalid-name
  """ReLU activation function with a threshold."""
  lambd = nn.functional.relu(lambd)
  return nn.functional.relu(x - lambd)


def softshrink(x, lambd):   # pylint: disable=invalid-name
  """Softshrink activation function with a threshold."""
  return nn.functional.relu(
    x - lambd) - nn.functional.relu(-x - lambd)
