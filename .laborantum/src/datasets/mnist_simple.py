import torchvision.datasets

from pathlib import Path
import torch

class MNISTSimpleDataset:
    def __init__(self, train=True):
        ...
        ## Load MNIST dataset here
        ## YOUR CODE HERE

        # In `__init__`, load MNIST with `torchvision.datasets.MNIST`.
        # Use the `train` argument to decide whether to load the training or validation split.
        # Store the raw images in `self.X` and labels in `self.y`.
        # Store the data under `~/` so that downloaded files do not clutter the repository.

        data = torchvision.datasets.MNIST(
            root=Path('~/').expanduser(),
            train=train,
            download=True,
        )

        self.X = data.data
        self.y = data.targets



    def __len__(self):
        res = 0
        ## Return number of items that is there in the dataset
        ## YOUR CODE HERE

        # In `__len__`, return the number of samples in the dataset.
        res = len(self.X)

        return res


    def __getitem__(self, index):
        sample = {}

        ## Return a sample of the dataset that corresponds to the input index
        ## YOUR CODE HERE

        # - In `__getitem__(self, index)`, return a dictionary with two keys:
        #   - `image`: the selected image converted to `float32` and scaled from `[0, 255]` to `[-1, 1]`;
        #   - `label`: the selected label converted to `long`.

        sample = dict(
            image = self.X[index].to(torch.float32) / 127.5 - 1.0,
            label = self.y[index].long(),
        )
        
        return sample 
