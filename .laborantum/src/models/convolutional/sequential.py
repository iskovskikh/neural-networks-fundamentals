import torch


class ResidualBottleneck(torch.nn.Module):
    def __init__(
            self, 
            in_channels,
            out_channels,
            prenormalization=lambda n_channels: torch.nn.Identity(),
            postnormalization=lambda n_channels: torch.nn.Identity(),
            activation=torch.nn.ReLU,
            compression=1,
            residual=True):

        super().__init__()
        self.block = torch.nn.Identity()
        self.bypass = torch.nn.Identity()
        self.residual = residual

        ## YOUR CODE HERE

        bottleneck_channels = max(out_channels // 4, 1)

        self.block = torch.nn.Sequential(
            prenormalization(in_channels),
            activation(),

            torch.nn.Conv2d(
                in_channels,
                bottleneck_channels,
                kernel_size=1,
                bias=False
            ),

            prenormalization(bottleneck_channels),
            activation(),

            torch.nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=compression,
                padding=1,
                bias=False
            ),

            prenormalization(bottleneck_channels),
            activation(),

            torch.nn.Conv2d(
                bottleneck_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            postnormalization(out_channels),
    )

        if residual:
            if in_channels == out_channels and compression == 1:
                self.bypass = torch.nn.Identity()
            else:
                self.bypass = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=compression,
                    bias=False
                )
        else:
            self.bypass = torch.nn.Identity()



    def forward(self, signal):
        ## YOUR CODE HERE

        out = self.block(signal)

        if self.residual:
            out = out + self.bypass(signal)
        
        return out


class FullyConvolutionalNN(torch.nn.Module):
    def __init__(
            self,
            block=lambda in_channels, out_channels: torch.nn.Conv2d(in_channels, out_channels, (1, 1),),
            in_channels=1,
            mid_channels=[16, 32, 64, 128],
            out_channels=10,
            n_blocks=[1, 1, 1, 1]):
        ...
        ## YOUR CODE HERE
        # Define network modules in the constructor

        super().__init__()

        levels = []

        current_channels = in_channels

        for level, channels in enumerate(mid_channels):

            for _ in range(n_blocks[level]):
                levels.append(block(current_channels, channels))
                current_channels = channels

            levels.append(
                block(current_channels, channels)
            )

            current_channels = channels

        self.encoder = torch.nn.Sequential(*levels)

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = torch.nn.Linear(
            current_channels,
            out_channels
        )



    def __forward_kernel(self, signal):
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward

        if signal.ndim == 3:
            signal = signal.unsqueeze(1)

        signal = self.encoder(signal)
        signal = self.pool(signal)
        signal = signal.flatten(1)
        signal = self.classifier(signal)


        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)

        # Put the result into the batch
        batch['signals'] = {'output': signal}

        # Perform postprocessing after we get the output
        self.postprocessing(batch)

        return batch

    def postprocessing(self, batch):

        # Take network's output from the batch
        signal = batch['signals']['output']

        ## YOUR CODE HERE
        signal = torch.argmax(signal, dim=1)

        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
