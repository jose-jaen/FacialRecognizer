from typing import Union, Optional

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from controllers.data_images import get_image_matrix


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.cumulative_explained_variance_: Optional[np.ndarray] = None
        self.scaler_: Optional[StandardScaler] = None

    @property
    def n_components(self) -> int:
        return self._n_components

    @n_components.setter
    def n_components(self, n_components: int) -> None:
        if not isinstance(n_components, int):
            incorrect = type(n_components).__name__
            raise TypeError(f"'n_components' must be 'int' but got '{incorrect}'")
        elif n_components <= 0:
            raise ValueError(f"'n_components' must be positive but got '{n_components}'")
        self._n_components = n_components

    def fit(self, data: Union[pl.DataFrame, np.ndarray]) -> None:
        """Apply PCA to a datamatrix where rows are observations.

        Args:
            data: Design matrix or matrix of features
        """
        if not isinstance(data, (pl.DataFrame, np.ndarray)):
            incorrect = type(data).__name__
            raise TypeError(
                f"'data' must be polards DataFrame or numpy array but got '{incorrect}'"
            )

        # Check for dimension
        if data.shape[1] < self._n_components:
            self._n_components = data.shape[1]

        # Scale data
        scaler = StandardScaler()
        scaler.fit(data)
        scaled = scaler.transform(data)
        self.scaler_ = scaler

        # Compute eigenvalues and eigenvectors
        small_cov = scaled @ scaled.T
        values, vectors = np.linalg.eig(small_cov)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indixes = np.argsort(a=values)[::-1]
        values = values[sorted_indixes]
        vectors = vectors[:, sorted_indixes]

        # Retain specified number of components
        values = values[:self._n_components]
        vectors = vectors[:, :self._n_components]

        # Get explained variance information
        self.explained_variance_ = values / np.sum(values)
        self.cumulative_explained_variance_ = np.cumsum(a=values) / np.sum(values)

        # Get eigenfaces
        eigenfaces = scaled.T @ vectors
        self.components_ = eigenfaces

    def visualize_eigenface(self, height: int, width: int) -> None:
        """Visualize an eigenface."""
        # Reshape color channels
        eigenface = self.components_[:, 23]
        size = height * width
        red = eigenface[:size].reshape((height, width))
        green = eigenface[size:2 * size].reshape((height, width))
        blue = eigenface[2 * size:].reshape((height, width))

        # Combine the components
        image = np.stack(arrays=(red, green, blue), axis=-1)

        # Visualize eigenface
        plt.imshow(image, interpolation='nearest')
        plt.show()
