import polars as pl

import optuna
from optuna.trial import Trial

from stats.fisherfaces import FisherFaces
from controllers.data_images import get_image_matrix


def objective(
        trial: Trial,
        train_data: pl.DataFrame,
        test_data: pl.DataFrame
) -> float:
    """Optimize the objective function using bayesian inference.

    Args:
        trial: Optuna study trial for bayesian inference
        train_data: Polars DataFrame with training data
        test_data: Polars DataFrame with validation data
    """
    space = {
      'n_neighbors': trial.suggest_int(name='n_neighbors', low=1, high=7),
      'threshold': trial.suggest_float(name='threshold', low=0.1, high=0.2),
      'var_explained': trial.suggest_float(
          name='var_explained',
          low=0.85,
          high=0.95
      )
    }

    # Set up classifier
    fisherfaces_clf = FisherFaces(**space, ignore_warnings=True)
    fisherfaces_clf.fit(train_data=train_data)
    predictions = fisherfaces_clf.predict(test_data=test_data)
    true_labels = test_data.select('masked_labels')['masked_labels'].to_numpy()
    accuracy = sum([i == j for i, j in zip(predictions, true_labels)])
    return accuracy / len(true_labels)


if __name__ == '__main__':
    # Get data and obtain labels
    df = get_image_matrix()
    train_labels = df.filter(pl.col('partition') == 'train') \
        .select('labels')['labels'].to_numpy()

    # Mask non-training instances
    df = df.with_columns(
        pl.col('labels')
        .map_elements(lambda x: x if x in train_labels else '0', return_dtype=str)
        .alias('masked_labels')
    )

    # Split training and validation data
    train = df.filter(pl.col('partition') == 'train')
    validation = df.filter(pl.col('partition') == 'validation')

    # Bayesian Hyperparameter tuning
    budget = 10
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='FisherFaces'
    )
    study.optimize(
        lambda trial: objective(trial, train, validation),
        n_trials=budget
    )

    # Update masked labels for full-train partition
    full_labels = df.filter(pl.col('partition') != 'test') \
        .select('labels')['labels'].to_numpy()
    df = df.with_columns(
        pl.col('labels')
        .map_elements(lambda x: x if x in full_labels else '0', return_dtype=str)
        .alias('masked_labels')
    )

    # Estimate model performance on unseen data
    full_train = df.filter(pl.col('partition') != 'test')
    test = df.filter(pl.col('partition') == 'test')
    test_labels = test.select('masked_labels')['masked_labels'].to_numpy()
    fisher_clf = FisherFaces(**study.best_params, ignore_warnings=True)
    fisher_clf.fit(full_train)
    preds = fisher_clf.predict(test_data=test)

    # Print results
    acc = sum([i == j for i, j in zip(preds, test_labels)]) / len(preds)
    print(f'Accuracy on test data: {100 * round(acc, 4)}%')
    print(f'Best hyperparameters: {study.best_params}')
