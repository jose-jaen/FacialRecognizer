# FacialRecognizer

A personal project leveraging both Statistical Learning and Deep Learning techniques to accurately recognize and classify faces. 

`PyTorch` and `polars` are used for efficiently handle the data in a distrubuted fashion. 

The <a target="_blank" rel="noopener noreferrer" href="https://cmp.felk.cvut.cz/~spacelib/faces/">`faces94`</a> dataset is utilized to train and evaluate models.

## Summary

<a target="_blank" rel="noopener noreferrer" href="https://cmp.felk.cvut.cz/~spacelib/faces/">`faces94`</a> consists of thousands of images, concretely around 20 images from more than 300 distinct individuals.

A facial recognition system is built with two well-known Statistical Learning methods implemented from scratch: <a href="https://www.face-rec.org/algorithms/PCA/jcn.pdf" target="_blank" rel="noopener noreferrer"><strong>Eigenfaces</strong></a> and 
<a href="https://cseweb.ucsd.edu/classes/wi14/cse152-a/fisherface-pami97.pdf" target="_blank" rel="noopener noreferrer"><strong>Fisherfaces</strong></a>. This system is able to discern whether a person belongs or not to the given database as well as its identification in case of the former.

Furthermore, an AI model (**Convolutional Neural Network**) is trained replicating the <a href="http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf" target="_blank" rel="noopener noreferrer"><strong>LeNet-5</strong></a> architecture using PyTorch as the preferred Deep Learning framework.

## Demo

Check the <a href="https://github.com/jose-jaen/FacialRecognizer/blob/main/demo.ipynb" rel="noreferrer noopener" target="_blank">Jupyter Notebook</a> showcasing the models with additional information about the algorithms.

## Requirements

The following software is needed to successfully run the project:

- `Python` >= 3.11
- `pip` >= 23.3.2

Optionally, if you own a GPU it is recommended to have `CUDA` installed.

For downloading and automatically managing the images it is necessary to use a Unix-based OS like Linux or macOS. A script that generalizes for Windows is being worked on.

Python libraries can be directly downloaded after creating a virtual environment with:
```bash
python3 -m pip install -r requirements.txt
```

## Getting Started
:warning: All Python scripts must be run from the speficic directory they are located :warning:

Firstly, it is needed to download and sort the images into three partitions:
```bash
python3 set_up_data.py
```

This will create a `data` folder with all the relevant images split into `train`, `validation` and `test` partitions.

After this you can train the models, for example, if we want to train the CNN:
```bash
python3 train_lenet.py
```

## Project Structure

### Data Engineering

```bash
FacialRecognizer/
├── config/
│   ├── .env
│   └── url.py
│
├── controllers/
│   ├── data_extractor.py
│   └── data_images.py
│     
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
└── set_up_data.py
```

The `config` folder simply stores the URL for the dataset within a variable in `url.py`. Since the URL was long, it was kept in an environmental variable.

Within `controllers`, the main data extraction and partition process is implemented: `set_up_data.py` serves as a `main` file for running the different methods of the `DataExtractor` class in `data_extractor.py`. `data_images.py` turns each image into an array of pixels stored in a polars dataframe for later processing.

Concretely, `faces94` is downloaded and a sample of females is guaranteed to be present in all datasets as to make sure the models generalize well. As to simulate the presence of individuals who do not belong to the original database, certain images are hidden from training in the folders `validation` and `test`.

### Statistical Learning

```bash
FacialRecognizer/
├── stats/
│   ├── pca.py
│   ├── lda.py
│   ├── knn.py
│   ├── eigenfaces.py
│   └── fisherfaces.py
│
├── train_eigenfaces.py
└── train_fisherfaces.py
```

`pca.py` includes the code for applying Principal Component Analysis (PCA) for dimensionality reduction. 

`lda.py` implements Fisher's Linear Discriminant Analysis (LDA) method for maximizing the spread between classes and minimizing the within-class variance.

A custom k-Nearest Neighbors (k-NN) algorithm can be found in `knn.py`. It chooses the most voted class among the selected candidates. The user may opt for one of the following statistical distances: 'cosine', 'seuclidean', 'euclidean', 'canberra'.

`eigenfaces.py` and `fisherfaces.py` define linear discriminants that feed the classifier defined in `knn.py`.

Both scripts `train_eigenfaces.py` and `train_fisherfaces.py` build a facial recognizer system based on the Eigenfaces and Fisherfaces approaches, tuning the hyperparameters with Bayesian Optimization.

### Deep Learning
```bash
FacialRecognizer/
├── deep_learning/
│   ├── dataset.py
│   └── lenet.py
│
└── train_lenet.py
```

`dataset.py` sets up a custom PyTorch Dataset structure to store the data and `lenet.py` lays out the code for defining LeNet-5 core architecture and its learning process. Finally, `train_lenet.py` fits a CNN to the data and performs classification, procuring the accuracy on test images.
