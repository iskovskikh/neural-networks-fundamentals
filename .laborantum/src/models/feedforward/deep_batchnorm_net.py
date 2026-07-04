import torch

class BatchNorm(torch.nn.Module):
    def __init__(self, beta=0.90, eps=1.0e-4):
        super().__init__()

        self.beta = beta
        self.eps = eps
        self.register_buffer('running_mean', None)
        
        ## YOUR CODE HERE

        self.register_buffer('running_var', None)


    def _init_stats(self, signal):
        channels = signal.shape[1]
        shape = [1, channels]

        self.running_mean = torch.zeros(
            shape,
            device=signal.device,
            dtype=signal.dtype,
        )

        self.running_var = torch.ones(
            shape,
            device=signal.device,
            dtype=signal.dtype,
        )

    def _check_stats(self, signal):
        return (
            self.running_mean is not None
            and self.running_mean.device == signal.device
            and self.running_mean.dtype == signal.dtype
            and self.running_mean.ndim == signal.ndim
            and self.running_mean.shape[1] == signal.shape[1]
        )

    def forward(self, signal):
        if not self._check_stats(signal):
            self._init_stats(signal)

        if self.training:
            ## YOUR CODE HERE
            
            mean = signal.mean(dim=0, keepdim=True)
            var = signal.var(dim=0, unbiased=False, keepdim=True)

            self.running_mean * (self.beta) + ((1 - self.beta) * mean)
            self.running_var * (self.beta) + ((1 - self.beta) * var)

            signal = (signal - mean) / torch.sqrt(var + self.eps)


            
        else:
            ## YOUR CODE HERE
            signal = (
                signal - self.running_mean
            ) / torch.sqrt(self.running_var + self.eps)


        return signal


class Residual(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        ## YOUR CODE HERE

        self.module = module


    def forward(self, signal):
        ## YOUR CODE HERE

        return signal + self.module(signal)


class Bottleneck(torch.nn.Module):
    def __init__(
            self, 
            in_channels,
            prenormalization=torch.nn.Identity,
            postnormalization=torch.nn.Identity,
            activation=torch.nn.ReLU,
            compression=1,
            **kwargs):

        super().__init__()
        self.block = torch.nn.Identity()

        ## YOUR CODE HERE

        hidden = in_channels // compression

        self.block = torch.nn.Sequential(
        prenormalization(),
        torch.nn.Linear(in_channels, hidden),
        activation(),
        torch.nn.Linear(hidden, in_channels),
        postnormalization(),
)

    def forward(self, signal):
        ## YOUR CODE HERE

        return signal



class DeepFullyConnectedNet(torch.nn.Module):
    def __init__(
            self,
            block=lambda n_channels: torch.nn.Linear(n_channels, n_channels),
            dim_input=28 * 28,
            dim_embed=128,
            dim_output=10,
            n_blocks=3):
        ...
        ## YOUR CODE HERE
        # Define network modules in the constructor
        super().__init__()

        # The design of this network is the following:
        # - it takes the input vector of dimensionality `dim_input`
        # - projects it to embedding space using an encoder
        #  (a simple linear transform `torch.nn.Linear`) of dimentinoality `dim_embed`

        self.encoder = torch.nn.Linear(dim_input, dim_embed)
        
        # - applies a sequence of `blocks` to the embedding

        hidden  = []
        for _ in range(n_blocks):
            hidden.append(block(dim_embed))

        self.blocks = torch.nn.Sequential(*hidden)

        # - after the last block is complete, applies a decoder
        #  (a simple linear transform `torch.nn.Linear`) to the dimentionality `dim_output`
        #  (that in the end is treated as the logits)

        self.decoder = torch.nn.Linear(dim_embed, dim_output)





    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward

        # - `__forward_kernel`, which receives an image tensor,
        #  flattens it from `[batch, 28, 28]` to `[batch, 784]`
        # , passes it through the blocks, and returns raw logits. 
        # This method is also used for the model visualization

        signal = self.encoder(signal)
        signal = self.blocks(signal)
        signal = self.decoder(signal)


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
