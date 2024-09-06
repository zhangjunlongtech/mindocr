import argparse
import gc
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from typing import Tuple

__all__ = ["CoreModel"]

class CoreModel(object):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def preprocess(self, **args):
        pass

    @abstractmethod
    def postprocess(self, args):
        pass

    @abstractmethod
    def load_model(self, args):
        pass
    
    @abstractmethod
    def infer(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass
