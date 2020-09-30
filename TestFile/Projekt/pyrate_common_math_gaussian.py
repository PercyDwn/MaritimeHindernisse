"""This module includes an abstraction of gaussian distributions."""

# Standard library
from dataclasses import dataclass

# Typing
from typing import cast

# Mathematics
from numpy import ndarray
from scipy.stats import multivariate_normal


@dataclass
class Gaussian:

    """A weighted gaussian distribution."""

    # ######################################
    # Dataclass members
    mean: ndarray
    covariance: ndarray
    weight: float = 1.0

    def __init__(self, mean: ndarray, covariance: ndarray, weight: float = 1.0):
        """Initialize a new gaussian distribution.

        Examples:
            >>> from numpy import array
            >>> from numpy import vstack
            >>> mean = vstack([0.0, 0.0])
            >>> covariance = array([[1.0, 0.0], [0.0, 1.0]])
            >>> N = Gaussian(mean, covariance)
            >>> N(vstack([0.0, 0.0]))
            0.15915494309189535

        Args:
            mean: The mean of the distribution as column vector
            covariance: The covariance matrix of the distribution
            weight: The weight of the distribution, e.g. within a mixture model
        """

        # Sanity checks on given parameters
        assert len(mean.shape) == 2 and mean.shape[1] == 1, "Mean needs to be a column vector!"
        assert len(covariance.shape) == 2, "Covariance needs to be a 2D matrix!"
        assert covariance.shape[0] == covariance.shape[1], "Covariance needs to be a square matrix!"
        assert covariance.shape[0] == mean.shape[0], "Dimensions of mean and covariance don't fit!"

        # Assign values
        self.mean = mean
        self.covariance = covariance
        self.weight = weight

    # ######################################
    # Properties following a common filter notation
    # pylint: disable=invalid-name
    @property
    def x(self) -> ndarray:
        """A common name in literature where the gaussian represents a state."""

        return self.mean

    @x.setter
    def x(self, value: ndarray):
        """A common name in literature where the gaussian represents a state."""

        self.mean = value

    @property
    def P(self) -> ndarray:
        """A common name in literature where the gaussian represents a state."""

        return self.covariance

    @P.setter
    def P(self, value: ndarray):
        """A common name in literature where the gaussian represents a state."""

        self.covariance = value

    @property
    def w(self) -> float:
        """A common name in literature where the gaussian represents a state."""

        return self.weight

    @w.setter
    def w(self, value: float):
        """A common name in literature where the gaussian represents a state."""

        self.weight = value

    def distribution(self) -> multivariate_normal:
    
      return multivariate_normal(mean=self.mean.T[0], cov=self.covariance)

    def distributionValue(self, distribution, value: ndarray) -> float:

      return self.weight * cast(float, distribution.pdf(value.T[0]))

    def __call__(self, value: ndarray) -> float:
        """Evaluate the gaussian at the given location."""

        # Compute weighted probability density function
        distribution = multivariate_normal(mean=self.mean.T[0], cov=self.covariance)

        return self.weight * cast(float, distribution.pdf(value.T[0]))

    def __eq__(self, other):
        return (
            (self.mean == other.mean).all()
            and (self.covariance == other.covariance).all()
            and self.weight == other.weight
        )


