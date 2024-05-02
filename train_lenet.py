import polars as pl

from torch.utils.data import DataLoader

from deep_learning.lenet import LenetTrain
from deep_learning.dataset import FacesDataset
from controllers.data_images import get_image_matrix

# Get image labels and partition
df = get_image_matrix(get_sobel=False)

# Map string labels to integers
unique_labels = df.select('labels').unique().to_series().to_list()
label_map = {label: idx for idx, label in enumerate(unique_labels)}
mapped_labels = df['labels'].replace(label_map).cast(pl.Int32)
df = df.with_columns(mapped_labels.alias('mapped_label'))
df = df.drop('labels')

# Obtain path to train and test images
train_path, test_path = [], []
for row in df.iter_rows(named=True):
    full_name = 'data/' + row['partition'] + '/' + row['image']
    if row['split'] == 'train':
        train_path.append(full_name)
    else:
        test_path.append(full_name)

# Create custom PyTorch data structures
trainset = FacesDataset(image_path=train_path, label_hash=label_map)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = FacesDataset(image_path=test_path, label_hash=label_map)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

if __name__ == '__main__':
    # Retrieve number of total labels
    n_labels = len(df['mapped_label'].unique())

    # Get image data
    sample_batch = next(iter(trainloader))
    sample_image = sample_batch['image']
    height, width = sample_image.shape[2:4]

    # Train LeNet-5 CNN
    lenet = LenetTrain(
        height=height,
        width=width,
        n_labels=n_labels,
        epochs=10,
        learning_rate=1e-3
    )
    lenet.train_loop(train_loader=trainloader, verbose=True)

    # Evaluate performance on test data
    accuracy = lenet.eval_performance(dataloader=testloader)
    print(f'Accuracy on test data: {accuracy}%')
