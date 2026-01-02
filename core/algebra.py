import torch
from abc import ABC, abstractmethod

class Algebra(ABC):
    @property
    @abstractmethod
    def dim(self):
        """Dimensionality of the algebra (e.g., 4 for Quaternion)"""
        pass

    @abstractmethod
    def expand_matrix(self, weights):
        """
        Takes a list of component weights [w_0, w_1, ...]
        Returns the constructed 'Real Constrained' matrix.
        """
        pass