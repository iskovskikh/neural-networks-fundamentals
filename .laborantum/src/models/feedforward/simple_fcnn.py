import torch

class SimpleFCNN(torch.nn.Module):
    def __init__(
            self,
            channels=None,
            n_classes=10,
            activation=torch.nn.ReLU
        ):

        ## YOUR CODE HERE
        # Define network modules in the constructor

        # - `__init__`, which creates the modules of the neural network.
        # Remember to initialize the parent `torch.nn.Module` with `super().__init__()`.
        super().__init__()


        if channels is None:
            channels = [128,]

        layers = []

        in_features = 28 * 28

        for out_features in channels:
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(activation())
            in_features = out_features

        layers.append(torch.nn.Linear(in_features, n_classes))

        self.network = torch.nn.Sequential(*layers)





    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward

        # - `__forward_kernel`, which receives an image tensor, flattens it from `[batch, 28, 28]` to `[batch, 784]`,
        # passes it through the fully connected layers, and returns raw logits.
        # This method is also used for the model visualization.

        signal = self.network(signal)

        return signal

    def forward(self, batch):
        signal = batch['data']['image']

        signal = self.__forward_kernel(signal)

        # Put the result into the batch
        batch['signals'] = {'output': signal}

        # Perform postprocessing after we get the output
        self.postprocessing(batch)

        # return batch['signals']['output']
        return batch

    def postprocessing(self, batch):

        # Take network's output from the batch
        signal = batch['signals']['output']

        ## YOUR CODE HERE

        # - `forward`, which receives the nested batch dictionary, reads `batch['data']['image']`, calls `__forward_kernel`, stores logits in `batch['signals']['output']`, calls `argmax` to get predicted classes, and stores them in `batch['postprocessed']['class']`.
        signal = torch.argmax(signal, dim=1)

        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
