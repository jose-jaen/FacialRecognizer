import os
import io
import random
import zipfile

import requests
import numpy as np

from config.url import website


class DataExtractor:
    def __init__(self, n_hidden: int = 25, n_test: int = 3):
        """Set up DataExtractor class for getting data.

        Args:
            n_hidden: Number of distinct individuals to leave out from training
            n_test: Number of test images to keep for each training instance
        """
        self.n_test = n_test
        self.n_hidden = n_hidden
        self.url: str = website

    @property
    def url(self) -> str:
        return self._url

    @url.setter
    def url(self, url: str) -> None:
        # Check valid datatype
        if not isinstance(url, str):
            raise TypeError(f"'url' must be 'str' but got '{type(url).__name__}'")

        # Check valid URL
        try:
            response = requests.head(url)
            if response.status_code != 200:
                raise ValueError(f'The provided URL is not working!')
        except requests.ConnectionError as e:
            raise e
        self._url = url

    @property
    def n_hidden(self) -> int:
        return self._n_hidden

    @n_hidden.setter
    def n_hidden(self, n_hidden: int) -> None:
        # Check valid datatype
        if not isinstance(n_hidden, int):
            raise TypeError(
                f"'n_hidden' must be 'int' but got '{type(n_hidden).__name__}'"
            )

        # Check valid value
        if n_hidden <= 0:
            raise ValueError(f'You must select at least one individual to hide!')
        elif n_hidden > 40:
            raise ValueError(f'You cannot choose more than 40 individuals to hide!')
        self._n_hidden = n_hidden

    @property
    def n_test(self) -> int:
        return self._n_test

    @n_test.setter
    def n_test(self, n_test: int) -> None:
        # Check valid datatype
        if not isinstance(n_test, int):
            raise TypeError(
                f"'test_size' must be 'int' but got '{type(n_test).__name__}'"
            )

        # Check valid value
        if n_test <= 0:
            raise ValueError(f"'n_test' must be strictly positive but got '{n_test}'")
        elif n_test >= 10:
            raise ValueError(f'Cannot take more than half of the data for testing!')
        self._n_test = n_test

    def get_data(self, reset: bool = False) -> None:
        """Download faces94 dataset and manage misclassified images.

        Args:
            reset: Whether to redownload the dataset or not
        """
        if os.path.isfile('faces94') and reset:
            os.system('rm -r faces94/')

        # Download dataset
        if not os.path.isfile('faces94'):
            download = requests.get(self._url)
            zip_file = zipfile.ZipFile(io.BytesIO(download.content))
            zip_file.extractall()
            print('Dataset has been successfully downloaded!')
        else:
            print('Dataset is already downloaded!')

    def set_up_data(self, random_state: int = 42) -> None:
        """Split the data into train / test partitions.

        Args:
            random_state: Seed value for reproducibility
        """
        random.seed(a=random_state)
        np.random.seed(seed=random_state)

        # Create directories for the images
        if not os.path.isfile('data'):
            os.system('mkdir data')
            os.system('mkdir data/train/')
            os.system('mkdir data/test/')

        # Check if user has the data
        if not os.path.isfile('faces94'):
            self.get_data()

        # Iterate over all directories
        os.chdir('faces94')
        directories = os.listdir()
        for main_directory in directories:
            os.chdir(main_directory)

            # Check every individual's directory
            people = os.listdir()

            # Get 28% of hidden images from women
            hidden_women = []
            n_women = round(0.28 * self._n_hidden)
            if main_directory == 'female':
                while len(hidden_women) < n_women:
                    candidate = people[random.randint(a=0, b=len(people) - 1)]
                    if candidate not in hidden_women:
                        hidden_women.append(candidate)

            # Get the rest from men
            hidden_male = []
            hidden_malestaff = []
            n_men = (self._n_hidden - n_women) // 2
            if main_directory == 'male':
                while len(hidden_male) < n_men:
                    candidate = people[random.randint(a=0, b=len(people) - 1)]
                    if candidate not in hidden_male:
                        hidden_male.append(candidate)
            elif main_directory == 'malestaff':
                while len(hidden_malestaff) < self._n_hidden - n_women - n_men:
                    candidate = people[random.randint(a=0, b=len(people) - 1)]
                    if candidate not in hidden_malestaff:
                        hidden_malestaff.append(candidate)

            # Get training / test distributions
            test_indixes = np.random.choice(a=20, size=self._n_test, replace=False)
            for individuals in people:
                os.chdir(individuals)
                images = [img for img in os.listdir() if img.endswith('.jpg')]
                total_hidden = hidden_malestaff + hidden_male + hidden_women
                for j in range(len(images)):
                    # Identify gender
                    female = main_directory == 'female'
                    if female:
                        os.system(f'mv {images[j]} female_{images[j]}')
                    else:
                        os.system(f'mv {images[j]} male_{images[j]}')

                    # Split images into partitions
                    new_image = f'female_{images[j]}' if female else f'male_{images[j]}'
                    if individuals in total_hidden or j in test_indixes:
                        os.system(f'mv {new_image} ../../../data/test/')
                    elif individuals not in total_hidden:
                        os.system(f'mv {new_image} ../../../data/train/')
                os.chdir('..')
            os.chdir('..')
        os.chdir('..')

        # Eliminate original dataset and isolate data directory
        os.system('rm -r faces94/')
        print('Train and test partitions were created!')

    def get_validation(self, n_valid: int) -> None:
        """Subdive train data into train and validation partitions.

        Args:
            n_valid: Number of distinct individuals to hide from the train set
        """
        if not os.path.isdir('data'):
            self.get_data()
            self.set_up_data(random_state=42)

        # Change directory
        os.chdir('data/train')

        # Get distinct male and female individuals
        individuals = [j.split('.')[0] for j in os.listdir()]
        female = [j for j in individuals if j.startswith('female_')]
        male = set(individuals).difference(set(female))

        # Select individuals to hide
        n_women = int(0.3 * n_valid)
        n_men = n_valid - n_women
        hidden_men = []
        hidden_women = []
        while len(hidden_women) != n_women:
            candidate = list(set(female))[random.randint(a=0, b=len(set(female)) - 1)]
            if candidate not in hidden_women:
                hidden_women.append(candidate)

        while len(hidden_men) != n_men:
            candidate = list(set(male))[random.randint(a=0, b=len(set(male)) - 1)]
            if candidate not in hidden_men:
                hidden_men.append(candidate)

        # Select the rest of validation images
        os.system('mkdir ../validation')
        for person in set(individuals):
            images = [j for j in os.listdir() if j.startswith(person)]
            if person in hidden_men + hidden_women:
                for image in images:
                    os.system(f'mv {image} ../validation/')
            else:
                sample = random.sample(population=images, k=2)
                for image in sample:
                    os.system(f'mv {image} ../validation/')
        print('Validation partition has been generated!')
