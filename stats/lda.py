from typing import Optional

import numpy as np
import polars as pl


class LinearDiscriminantAnalysis:
    def __init__(self, var_explained: float = 0.9):
        self.var_explained = var_explained
        self.coef_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.cumulative_explained_variance_: Optional[np.ndarray] = None

    @property
    def var_explained(self) -> float:
        return self._var_explained

    @var_explained.setter
    def var_explained(self, var_explained: float) -> None:
        if not isinstance(var_explained, float):
            incorrect = type(var_explained).__name__
            raise TypeError(f"'var_explained' must be 'float' but got '{incorrect}'")
        elif var_explained <= 0.0:
            raise ValueError(
                f"'var_explained' must be strictly positive but got '{var_explained}'"
            )
        elif var_explained > 1.0:
            raise ValueError(
                f"'var_explained' cannot be greater than 1.0 but got '{var_explained}'"
            )
        self._var_explained = var_explained

    def fit(self, data: pl.DataFrame) -> None:
        """Fit a LDA classifier to the data to maximize separation between classes.

        Args:
            data: Design / feature matrix to fit the classifer
        """
        if not isinstance(data, pl.DataFrame):
            incorrect = type(data).__name__
            raise TypeError(f"'data' must be polars DataFrame but got '{incorrect}'")

        # Compute overall and class averages
        avg_face = data.drop('labels').mean().to_numpy()
        avg_class_face = data.group_by(by='labels', maintain_order=True) \
            .agg(pl.col('*').mean())

        # Get covariance matrices
        within = 0
        between = 0
        total_labels = data.unique(
            subset='labels',
            maintain_order=True
        ).select('labels')
        for label in total_labels['labels']:
            class_data = data.filter(pl.col('labels') == label).drop('labels')
            n_class = len(data.filter(pl.col('labels') == label))

            # Update within-class covariance
            class_within = np.cov(class_data.to_numpy().T)
            within += (n_class - 1) * class_within

            # Update between-class covariance
            avg_class_between = avg_class_face.filter(
                pl.col('labels') == label
            ).drop('labels')
            class_between = avg_class_between.to_numpy() - avg_face
            between += n_class * (class_between.T @ class_between)

        # Apply optimization criterion
        criterion = np.linalg.inv(within) @ between
        values, vectors = np.linalg.eig(criterion)

        # Sort diagonalization results
        sorted_indixes = np.argsort(a=values)[::-1]
        values = values[sorted_indixes]
        vectors = vectors[:, sorted_indixes]

        # Get explained variance information
        self.explained_variance_ = values / np.sum(values)
        self.cumulative_explained_variance_ = np.cumsum(a=values) / np.sum(values)

        # Retain specified number of components
        indx = np.argwhere(self.cumulative_explained_variance_ >= self._var_explained)
        self.coef_ = vectors[:, :int(indx[0][0]) + 1]
