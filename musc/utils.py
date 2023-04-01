'''Shared utility functions'''
from pathlib import Path


def get_musc_root() -> Path:
    '''Get the root dir of the project'''
    return Path(__file__).parent.parent
