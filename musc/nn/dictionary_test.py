"""Test dictionary"""
# import unittest
import torch
from musc.nn.dictionary import UDictionary, TransposedUDictionary

COARSEST_ATOM_SIDELEN = 8
BATCH_SIZE = 2


class TestUDictionary:
  """Test the U dictionary"""
  def test_transpose(self, kernel_size, hidden_layer_widths, n_classes):
    """Test the transpose"""
    x_list = []
    atom_sidelen = COARSEST_ATOM_SIDELEN
    for width in hidden_layer_widths:
      x_list.append(
          torch.rand(BATCH_SIZE, width, atom_sidelen, atom_sidelen)
      )
      atom_sidelen *= 2

    u_dictionary = UDictionary(
        kernel_size=kernel_size,
        hidden_layer_widths=hidden_layer_widths,
        n_classes=n_classes)

    # output produced by the dictionary_model model
    dictionary_out = u_dictionary(x_list)

    # randomly initialize a tensor of the same shape
    rand_out = torch.rand_like(dictionary_out)

    # compute the adjoint of that model
    trans_dictionary = TransposedUDictionary(u_dictionary)

    trans_dictionary_out = trans_dictionary(rand_out)

    inner_prod_dictionary_out = torch.dot(
        dictionary_out.flatten(),
        rand_out.flatten())

    inner_prod_dictionary_in = 0
    for x_in, x_trans_out in list(zip(x_list, trans_dictionary_out)):
      inner_prod_dictionary_in += torch.dot(
          x_in.flatten(),  x_trans_out.flatten())

    torch.testing.assert_close(
        inner_prod_dictionary_out,
        inner_prod_dictionary_in,
        msg="incorrect adjoint")


if __name__ == '__main__':
  test = TestUDictionary()
  test.test_transpose(
      kernel_size=3,
      hidden_layer_widths=[1024, 512, 256, 128, 64],
      n_classes=3
      )
  test.test_transpose(
      kernel_size=3,
      hidden_layer_widths=[512, 256, 128, 64, 32],
      n_classes=3
      )
  test.test_transpose(
      kernel_size=3,
      hidden_layer_widths=[512, 256, 128, 64, 32],
      n_classes=1
      )
  test.test_transpose(
      kernel_size=5,
      hidden_layer_widths=[512, 256, 128, 64, 32],
      n_classes=1
      )
  test.test_transpose(
      kernel_size=7,
      hidden_layer_widths=[512, 256, 128, 64, 32],
      n_classes=1
      )
  print('all tests passed.')
