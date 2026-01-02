import torch
from abc import ABC, abstractmethod

class Algebra(ABC):
    @property
    @abstractmethod
    def dim(self):
        """Number of learnable parameters per connection."""
        pass

    @property
    def mat_dim(self):
        """
        The size of the square matrix block generated.
        By default, equal to dim (works for Complex, Quat).
        Override this for Lie Algebras like SO(n).
        """
        return self.dim

    @abstractmethod
    def expand_matrix(self, weights):
        pass