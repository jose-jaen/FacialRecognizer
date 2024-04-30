import warnings
from typing import Optional, NoReturn, List

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from stats.pca import PCA
from stats.knn import KNeighborsClassifier
from stats.lda import LinearDiscriminantAnalysis


class FisherFaces:
    def __init__(
            self,
            n_neighbors: int,
            var_explained: float,
            threshold: float,
            ignore_warnings: bool = False
    ):
        """Set up FisherFaces classifier.

        Args:
            n_neighbors: Number of neighbors for majority voting
            threshold: Value to determine when a face does not belong to dataset
            var_explained: Target percentage of explained variance
            ignore_warnings: Whether to ignore potential complex numbers warnings
        """
        self.n_neighbors = n_neighbors
        self.var_explained = var_explained
        self.threshold = threshold

        # Whether to ignore potential complex number warnings or not
        self.ignore_warnings = ignore_warnings

        # Training labels / identifiers
        self.labels: Optional[np.ndarray] = None

        # Scaler object to avoid data leakage
        self.scaler_: Optional[StandardScaler] = None

        # Selected largest eigenvectors
        self.eigenfaces: Optional[np.ndarray] = None

        # Weight matrix from LDA
        self.lda_coef_: Optional[np.ndarray] = None

        # LDA projection of Fisherfaces
        self.fisher_projection: Optional[np.ndarray] = None

    def fit(self, train_data: pl.DataFrame) -> NoReturn:
        """Fit a LDA classifier to the data to maximize separation between classes.

        Args:
            train_data: Polars DaatFrame with
        """
        if self.ignore_warnings:
            warnings.filterwarnings('ignore')

        # Store training labels
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
        projection = pl.from_numpy(scaled_train @ self.eigenfaces)
        projection = projection.with_columns(
            pl.Series('labels', list(self.labels), dtype=pl.String)
        )

        # Apply LDA on projected data
        lda = LinearDiscriminantAnalysis(var_explained=self.var_explained)
        lda.fit(projection)
        fisher_proj = projection.drop('labels').to_numpy() @ lda.coef_
        self.lda_coef_ = lda.coef_
        self.fisher_projection = fisher_proj

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
        knn.fit(data=self.fisher_projection, labels=self.labels)

        # Obtain predictions from test data
        for row in test_data.iter_rows():
            scaled_instance = self.scaler_.transform(np.array(row[:-3]).reshape(1, -1))
            projected_instance = scaled_instance @ self.eigenfaces
            fisher_instance = projected_instance @ self.lda_coef_
            prediction = knn.predict(image=fisher_instance)
            predictions.append(prediction)
        return predictions
