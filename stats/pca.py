from typing import Union, Optional

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


class PCA:
    def __init__(self, var_explained: float = 0.9):
        """Extract principal componenets of the data matrix.

        Args:
            var_explained: Target percentage of explained variance
        """
        self.var_explained = var_explained
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.cumulative_explained_variance_: Optional[np.ndarray] = None
        self.scaler_: Optional[StandardScaler] = None

    @property
    def var_explained(self) -> float:
        return self._var_explained

    @var_explained.setter
    def var_explained(self, var_explained: float) -> None:
        if not isinstance(var_explained, float):
            incorrect = type(var_explained).__name__
            raise TypeError(f"'var_explained' must be 'int' or 'float' but got '{incorrect}'")
        elif var_explained <= 0.0:
            raise ValueError(f"'var_explained' must be positive but got '{var_explained}'")
        elif var_explained > 1.0:
            raise ValueError(
                f"'var_explained' cannot be greater than 1.0 but got '{var_explained}'"
            )
        self._var_explained = var_explained

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

        # Convert to numpy array for scaling (avoid feature names warning)
        elif isinstance(data, pl.DataFrame):
            data = data.to_numpy()

        # Scale data
        scaler = StandardScaler()
        scaler.fit(data)
        scaled = scaler.transform(data)
        self.scaler_ = scaler

        # Compute eigenvalues and eigenvectors
        small_cov = (1 / (data.shape[0] - 1)) * scaled @ scaled.T
        values, vectors = np.linalg.eig(small_cov)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indixes = np.argsort(a=values)[::-1]
        values = values[sorted_indixes]
        vectors = vectors[:, sorted_indixes]

        # Get explained variance information
        self.explained_variance_ = values / np.sum(values)
        self.cumulative_explained_variance_ = np.cumsum(a=values) / np.sum(values)

        # Retain specified number of components
        indx = np.argwhere(self.cumulative_explained_variance_ >= self._var_explained)
        vectors = vectors[:, :int(indx[0][0]) + 1]

        # Get eigenfaces
        self.components_ = scaled.T @ vectors

    def visualize_eigenface(self, height: int, width: int, face: int = 23) -> None:
        """Visualize an eigenface."""
        if not isinstance(face, int):
            raise TypeError(f"'face' must be 'int' but got '{type(face).__name__}'")

        # Check validity of sample eigenface to show
        if face < 0:
            raise ValueError(f"'face' cannot be negative but got '{face}'")
        elif face >= self.components_.shape[1]:
            face = self.components_.shape[1] - 1

        # Reshape color channels
        eigenface = self.components_[:, face]
        image = eigenface.reshape(height, width)

        # Visualize eigenface
        plt.imshow(image, cmap='viridis')
        plt.title('Eigenface visualization')
        plt.colorbar()
        plt.show()
