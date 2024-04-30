import os

import cv2
import numpy as np
import polars as pl
from PIL import Image


def get_image_matrix(get_sobel: bool = True) -> pl.DataFrame:
    """Get image pixels and labels in a polars DataFrame.

    Args:
        get_sobel: Whether to apply edge detection or not
    """
    split = []
    matrix = []
    labels = []
    label_dict = {}
    full_images = []

    # Change directory to where data is located
    os.chdir('../data')
    for partition in ['train', 'validation', 'test']:
        os.chdir(partition)
        images = [i for i in os.listdir() if i.endswith('.jpg')]
        full_images.extend(images)
        for img in images:
            if get_sobel:
                # Convert image to grayscale
                pic = Image.open(img).convert('L')
                pic_numpy = np.array(pic)

                # Normalize the image
                pic_normalized = pic_numpy / 255

                # Apply Sobel edge detection
                sobel_x = cv2.Sobel(pic_normalized, cv2.CV_64F, dx=1, dy=0, ksize=3)
                sobel_y = cv2.Sobel(pic_normalized, cv2.CV_64F, dx=0, dy=1, ksize=3)

                # Combine and normalize the Sobel images
                sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                sobel_combined = sobel_combined / np.max(sobel_combined)

                # Flatten the Sobel images
                sobel_flattened = sobel_combined.flatten()

                # Append features to the matrix
                matrix.append(sobel_flattened)

            # Get labels and set up image matrix
            label = img.split('.')[0].split('_')[1]
            if not get_sobel:
                if label not in label_dict:
                    label_dict[label] = 0
                else:
                    label_dict[label] += 1

                # Get to train
                if label_dict[label] < 15:
                    split.append('train')
                else:
                    split.append('test')

            labels.append(label)
        os.chdir('..')

    # Retrieve partitions
    os.chdir('..')
    test_imgs = [i for i in os.listdir('data/test') if i.endswith('.jpg')]
    train_imgs = [i for i in os.listdir('data/train') if i.endswith('.jpg')]
    valid_imgs = [i for i in os.listdir('data/validation') if i.endswith('.jpg')]
    partition = ['train'] * len(train_imgs) + ['validation'] * len(valid_imgs)
    partition += ['test'] * len(test_imgs)

    # Set up polars dataframe and sort by labels
    if get_sobel:
        matrix = np.array(matrix)
        df = pl.from_numpy(matrix)
        df = df.with_columns(
            [
                pl.Series(name='labels', values=labels),
                pl.Series(name='partition', values=partition)
            ]
        )
    else:
        df = pl.DataFrame(
            [
                pl.Series(name='image', values=full_images),
                pl.Series(name='labels', values=labels),
                pl.Series(name='partition', values=partition),
                pl.Series(name='split', values=split)
            ]
        )
    df = df.sort(pl.col('labels'))
    return df
