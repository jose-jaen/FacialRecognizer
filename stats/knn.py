from statistics import mode
from typing import Union, Optional, List

import numpy as np
import polars as pl

from scipy.spatial.distance import cdist


class KNeighborsClassifier:
    def __init__(
            self,
            n_neighbors: int,
            threshold: Union[int, float],
            distance: str
    ):
        """Classify images based on k-nearest neighbors.

        Args:
            n_neighbors: Number of neighbors for majority voting
            threshold: Value to determine when a face does not belong to dataset
            distance: Statistical metric to use to compare images
        """
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.distance = distance

        # Design / feature matrix to fit the classifier
        self.data: Optional[np.ndarray] = None

        # Training instance labels
        self.labels: Optional[np.ndarray] = None

    @property
    def n_neighbors(self) -> int:
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, n_neighbors: int) -> None:
        if not isinstance(n_neighbors, int):
            incorrect = type(n_neighbors).__name__
            raise TypeError(f"'n_neighbors' must be 'int' but got '{incorrect}'")
        elif n_neighbors <= 0:
            raise ValueError(
                f"'n_neighbors' must be strictly positive but got '{n_neighbors}'"
            )
        self._n_neighbors = n_neighbors

    @property
    def threshold(self) -> Union[int, float]:
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: Union[int, float]) -> None:
        if not isinstance(threshold, (int, float)):
            incorrect = type(threshold).__name__
            raise TypeError(f"'threshold' must be 'int' or 'float' but got '{incorrect}'")
        elif threshold < 0:
            raise ValueError(
                f"Statistical distances cannot be negative but got '{threshold}'"
            )
        self._threshold = threshold

    @property
    def distance(self) -> str:
        return self._distance

    @distance.setter
    def distance(self, distance: str) -> None:
        valid_distances = ['cosine', 'seuclidean', 'euclidean', 'canberra']
        if not isinstance(distance, str):
            raise TypeError(
                f"'distance' must be 'str' but got '{type(distance).__name__}'"
            )
        elif distance not in valid_distances:
            raise ValueError(
                f"'distance' must be in '{valid_distances}' but got '{distance}'"
            )
        self._distance = distance

    def fit(
            self,
            data: Union[pl.DataFrame, np.ndarray],
            labels: Union[List[str], np.ndarray]
    ) -> None:
        """Fit the k-nearest neighbors classifier from the training dataset.

        Args:
            data: Design / feature matrix to fit the classifier
            labels: Identification of training individuals
        """
        if not isinstance(data, (pl.DataFrame, np.ndarray)):
            incorrect = type(data).__name__
            raise TypeError(
                f"'data' must be polars DataFrame or numpy array but got '{incorrect}'"
            )
        elif isinstance(data, pl.DataFrame):
            data = data.to_numpy()

        # Check labels validity
        if not isinstance(labels, (list, np.ndarray)):
            raise TypeError(
                f"'labels' must be numpy array but got '{type(labels.__name__)}'"
            )
        elif isinstance(labels, list):
            labels = np.array(labels)

        # Set data matrix and labels
        self.data = data
        self.labels = labels

    def predict(self, image: np.ndarray) -> str:
        """Get facial recognition predictions for a test image.

        Args:
            image: Test photo to classify

        Returns:
            prediction: Training identifier if face is recognized, zero otherwise
        """
        prediction = '0'

        # Compute statistical distance matrix
        distance = cdist(XA=self.data, XB=image, metric=self._distance)

        # Get closest training instance if threshold is not met
        if np.min(distance.T) < self._threshold:
            neighbors = np.argsort(a=distance.T)[0][:self._n_neighbors]
            prediction = mode(self.labels[neighbors])
        return prediction
