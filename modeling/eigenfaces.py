from typing import Optional, NoReturn, List

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from stats.pca import PCA
from stats.knn import KNeighborsClassifier


class EigenFaces:
    def __init__(
            self,
            n_neighbors: int,
            var_explained: float,
            threshold: float
    ):
        """Set up FisherFaces classifier.

        Args:
            n_neighbors: Number of neighbors for majority voting
            threshold: Value to determine when a face does not belong to dataset
            var_explained: Target percentage of explained variance
        """
        self.n_neighbors = n_neighbors
        self.var_explained = var_explained
        self.threshold = threshold

        # Training labels / identifiers
        self.labels: Optional[np.ndarray] = None

        # Scaler object to avoid data leakage
        self.scaler_: Optional[StandardScaler] = None

        # Selected largest eigenvectors
        self.eigenfaces: Optional[np.ndarray] = None

        # Projected data into the face space
        self.projection: Optional[np.ndarray] = None

    def fit(self, train_data: pl.DataFrame) -> NoReturn:
        """Fit a PCA-based classifier to reduce the dimensionality of the data.

        Args:
            train_data: Polars DaatFrame with
        """
        self.labels = train_data.select(pl.col('labels'))['labels'].to_numpy()

        # Apply PCA, get scaler and eigenfaces
        pca = PCA(var_explained=self.var_explained)
        pca.fit(train_data.drop(train_data.columns[-3:]))
        self.scaler_ = pca.scaler_
        self.eigenfaces = pca.components_

        # Project data into the face space and add labels
        scaled_train = self.scaler_.transform(
            train_data.drop(train_data.columns[-3:]).to_numpy()
        )
        self.projection = scaled_train @ self.eigenfaces

    def predict(self, test_data: pl.DataFrame) -> List[str]:
        """Predict labels for test data.

        Args:
            test_data: Polars DataFrame with test data
        Returns:
            predictions: Predicted labels
        """
        predictions = []

        # Instantiate k-NN classifier
        knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            threshold=self.threshold,
            distance='cosine'
        )
        knn.fit(data=self.projection, labels=self.labels)

        # Obtain predictions from test data
        for row in test_data.iter_rows():
            scaled_instance = self.scaler_.transform(np.array(row[:-3]).reshape(1, -1))
            projected_instance = scaled_instance @ self.eigenfaces
            prediction = knn.predict(image=projected_instance)
            predictions.append(prediction)
        return predictions
