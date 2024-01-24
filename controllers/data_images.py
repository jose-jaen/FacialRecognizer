import os

import numpy as np
import polars as pl
from PIL import Image


def get_image_matrix() -> pl.DataFrame:
    """Get image pixels and labels in a polars DataFrame."""
    matrix = []
    labels = []

    # Change directory to where data is located
    os.chdir('../data')
    for partition in ['train', 'validation','test']:
        os.chdir(partition)
        images = [i for i in os.listdir() if i.endswith('.jpg')]
        for img in images:
            # Exract red, green and blue color channels
            pic = Image.open(img)
            pic = np.array(pic)
            red = np.array(pic[:, :, 0].flatten())
            green = np.array(pic[:, :, 1].flatten())
            blue = np.array(pic[:, :, 2].flatten())
            vector = np.concatenate((red, green, blue))

            # Get labels and set up image matrix
            labels.append(img.split('.')[0])
            matrix.append(vector)
        os.chdir('..')

    # Retrieve partitions
    os.chdir('..')
    test_imgs = [i for i in os.listdir('data/test') if i.endswith('.jpg')]
    train_imgs = [i for i in os.listdir('data/train') if i.endswith('.jpg')]
    valid_imgs = [i for i in os.listdir('data/validation') if i.endswith('.jpg')]
    partition = ['train']*len(train_imgs) + ['validation']*len(valid_imgs)
    partition += ['test']*len(test_imgs)

    # Set up polars dataframe
    matrix = np.array(matrix)
    df = pl.from_numpy(matrix)
    df = df.with_columns(
        [
            pl.Series(name='labels', values=labels),
            pl.Series(name='partition', values=partition)
        ]
    )
    return df
