import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


class LeNet5(nn.Module):
    def __init__(self, height: int, width: int, n_labels: int):
        """Set up LeNet-5 architecture.

        Args:
            height: Height of the images in pixels
            width: Width of the images in pixels
            n_labels: Number of distinct individuals to predict
        """
        super().__init__()
        self.height = height
        self.width = width
        self.n_labels = n_labels

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            padding=0
        )

        # Max pool layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dimensions for the FC layer
        self.final_height = int(((self.height - 4) / 2 - 4) / 2)
        self.final_width = int(((self.width - 4) / 2 - 4) / 2)

        # Linear layers
        self.linear1 = nn.Linear(
            in_features=16 * self.final_height * self.final_width,
            out_features=120
        )
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=self.n_labels)

        # Activation functions
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward pass for Lenet5."""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten the tensor for FC layers
        x = x.view(x.shape[0], 16 * self.final_height * self.final_width)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.logsoftmax(self.linear3(x))
        return x


class LenetTrain(LeNet5):
    def __init__(
            self,
            height: int,
            width: int,
            n_labels: int,
            epochs: int = 10,
            learning_rate: float = 0.001
    ):
        """Set up the learning process for LeNet-5.

        Args:
            epochs: Number of epochs for learning
            learning_rate: Stepsize for learning
        """
        super().__init__(height, width, n_labels)
        self.epochs = epochs
        self.criterion = nn.NLLLoss()
        self.learning_rate = learning_rate
        self.optim = optim.Adam(self.parameters(), self.learning_rate)

        # Keep track of training loss
        self.loss_during_training = []

        # Use GPU if possible
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def train_loop(self, train_loader: DataLoader, verbose: bool = False) -> None:
        """Train Lenet5 for the given number of epochs.

        Args:
            train_loader: PyTorch DataLoader with training data
            verbose: Whether to provide training information or not
        """
        for epoch in range(int(self.epochs)):
            running_loss = 0
            for image_batch in train_loader:
                # Move data to GPU if possible
                images = image_batch['image'].to(self.device)
                labels = image_batch['label'].to(self.device)

                # Reset gradients for each batch
                self.optim.zero_grad()

                # Forward pass for each batch
                out = self.forward(images)

                # Update loss function
                loss = self.criterion(out, labels)
                running_loss += loss.item()

                # Backpropagation
                loss.backward()
                self.optim.step()

            # Keep track of the loss function
            self.loss_during_training.append(running_loss / len(train_loader))
            if verbose:
                current_loss = round(self.loss_during_training[-1], 4)
                print(
                    f'Epoch {epoch + 1}:\nTraining loss: {current_loss}'
                )

    def eval_performance(self, dataloader: DataLoader) -> float:
        """Assess performance on test data.

        Args:
            dataloader: PyTorch dataloader with test data
        """
        accuracy = 0

        # Freeze gradients
        with torch.no_grad():
            for image_batch in dataloader:
                # Move to GPU if possible
                images = image_batch['image'].to(self.device)
                labels = image_batch['label'].to(self.device)

                # Estimate probabilities and keep the highest one
                probs = self.forward(images)
                _, top_class = probs.topk(k=1, dim=1)
                top_class = torch.tensor([i[0] for i in top_class]).to(self.device)

                # Compare estimations with true labels
                equals = top_class == labels
                accuracy += torch.mean(equals.clone().detach().type(torch.FloatTensor))
        accuracy /= len(dataloader)
        return round(100 * accuracy.item(), 4)
