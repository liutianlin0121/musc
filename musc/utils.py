'''Shared utility functions'''
from pathlib import Path


def get_musc_root() -> Path:
  '''Get the root dir of the project'''
  return Path(__file__).parent.parent

def remove_frame(ax):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()
