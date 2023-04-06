"""Multiscale U-Net like dictionary"""
import torch.nn as nn
import torch.nn.functional as F
import torch


def conv2d_to_convtranspose2d(conv2d_op):
  """Convert a nn.Conv2D instance to nn.ConvTranspose2d instance
  Args:
      conv2d_op: nn.Conv2D instance

  Returns:
      transposed_conv2d: transpose of the Conv2D instance
  """
  # transpose convolution
  assert conv2d_op.bias is None, "Need bias-free Conv2D operator."
  assert conv2d_op.groups == 1, "Need groups=1"
  assert conv2d_op.padding_mode == "zeros", "Need zero padding"

  transposed_conv2d = nn.ConvTranspose2d(
      in_channels=conv2d_op.out_channels,
      out_channels=conv2d_op.in_channels,
      kernel_size=conv2d_op.kernel_size,
      stride=conv2d_op.stride,
      padding=conv2d_op.padding,
      dilation=conv2d_op.dilation,
      bias=False)

  # tie the weights of transpose convolution with convolution
  transposed_conv2d.weight = conv2d_op.weight
  return transposed_conv2d


def convtranspose2d_to_conv2d(convtranspose2d_op):
  """Convert a nn.ConvTranspose2d instance to a nn.Conv2D instance
  Args:
      convtranspose2d_op: nn.ConvTranspose2d instance

  Returns:
      conv2d_op: transpose of the Conv2D instance
  """
  # transpose convolution
  assert convtranspose2d_op.bias is None, "Need bias-free Conv2D operator."
  assert convtranspose2d_op.groups == 1, "Need groups=1"
  assert convtranspose2d_op.padding_mode == "zeros", "Need zero padding"

  conv2d_op = nn.Conv2d(
      in_channels=convtranspose2d_op.out_channels,
      out_channels=convtranspose2d_op.in_channels,
      kernel_size=convtranspose2d_op.kernel_size,
      stride=convtranspose2d_op.stride,
      padding=convtranspose2d_op.padding,
      dilation=convtranspose2d_op.dilation,
      bias=False)

  # tie the weights of transpose convolution with convolution
  conv2d_op.weight = convtranspose2d_op.weight
  return conv2d_op


class UpBlock(nn.Module):
  """ The composition of a ConvTranspose2d and a Conv2d functions. """
  def __init__(self,
               kernel_size: int,
               in_channels: int,
               out_channels: int):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    # the up-sampling operation
    self.up_sampling = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        bias=False)

    # the convolution operation
    self.conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        bias=False)

  def forward(self, x1, x2):
    # pylint: disable=missing-function-docstring
    # pylint:disable=invalid-name
    x1 = self.up_sampling(x1)

    diff_y = x2.size()[2] - x1.size()[2]
    diff_x = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2])

    # input is CHW
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)


class TransposedUpBlock(nn.Module):
  """Transpose of UpBlock module. """

  def __init__(self, up_block_model):
    super().__init__()
    self.convtranspose2d_op = conv2d_to_convtranspose2d(
        up_block_model.conv)

    self.conv2d_op = convtranspose2d_to_conv2d(
        up_block_model.up_sampling)

  def forward(self, x):
    # pylint: disable=missing-function-docstring
    # pylint:disable=invalid-name
    x = self.convtranspose2d_op(x)
    half_num_channels = int(x.shape[1]/2)
    # input is CHW
    x2 = x[:, :half_num_channels, :, :]
    x1 = x[:, half_num_channels:, :, :]
    x1 = self.conv2d_op(x1)
    return (x1, x2)


class UDictionary(nn.Module):
  """U-Net like dictionary. """
  def __init__(self,  kernel_size, hidden_layer_widths, n_classes):
    super().__init__()

    self.hidden_layer_widths = hidden_layer_widths
    self.n_classes = n_classes

    in_out_list = [[hidden_layer_widths[i], hidden_layer_widths[i+1]]
                    for i in range(len(hidden_layer_widths)-1)]

    # the initial convolution
    self.initial_conv = nn.Conv2d(hidden_layer_widths[0],
                                  hidden_layer_widths[0],
                                  kernel_size=kernel_size,
                                  padding=kernel_size // 2,
                                  bias=False)

    # the up blocks
    up_blocks = []
    for in_out_dims in in_out_list:
      new_up_block = UpBlock(kernel_size, *in_out_dims)
      up_blocks.append(new_up_block)
    self.up_blocks = nn.ModuleList(up_blocks)

    # the output convolution
    self.out_conv = nn.Conv2d(hidden_layer_widths[-1],
                              n_classes,
                              kernel_size=1,
                              bias=False)

  def forward(self, x_list):
    # pylint: disable=missing-function-docstring
    # pylint:disable=invalid-name
    # x_list is ordered from wide-channel to thin-channel.
    num_res_levels = len(x_list)

    x_prev = self.initial_conv(x_list[0])

    for i in range(1, num_res_levels):
      x = x_list[i]
      up_block = self.up_blocks[i-1]
      x_prev = up_block(x_prev, x)

    out = self.out_conv(x_prev)
    return out


class TransposedUDictionary(nn.Module):
  """Transposed U-Net like dictionary. """
  def __init__(self, dictionary_model):
    super().__init__()

    self.transposed_out_conv = conv2d_to_convtranspose2d(
        dictionary_model.out_conv)
    self.transposed_initial_conv = conv2d_to_convtranspose2d(
        dictionary_model.initial_conv)

    adjoint_up_blocks = []
    for up_block in dictionary_model.up_blocks:
      # prepend the upblock
      adjoint_up_blocks.insert(0, TransposedUpBlock(up_block))

    self.adjoint_up_blocks = nn.ModuleList(adjoint_up_blocks)

  def forward(self, y):
    # pylint: disable=missing-function-docstring
    # pylint:disable=invalid-name
    y = self.transposed_out_conv(y)
    x_list = []

    for adjoint_up_block in self.adjoint_up_blocks:
      y, x = adjoint_up_block(y)
      x_list.insert(0, x)

    y = self.transposed_initial_conv(y)
    x_list.insert(0, y)
    return x_list


def compute_power_iterations(u_dictionary, num_iterations: int):
  D = u_dictionary
  Dt = TransposedUDictionary(u_dictionary)
  # we compute the spectral radius of the matrix D D^{\top}
  DDt = lambda x: D(Dt(x))

  b_k = torch.randn(1, 1, 128, 128)

  for _ in range(num_iterations):
      # calculate the matrix-by-vector product Ab
      b_k1 = DDt(b_k)

      # calculate the norm
      b_k1_norm = torch.norm(b_k1)

      # re normalize the vector
      b_k = b_k1 / b_k1_norm
  return torch.dot(b_k1.flatten(), b_k.flatten()) / torch.dot(b_k.flatten(), b_k.flatten())

