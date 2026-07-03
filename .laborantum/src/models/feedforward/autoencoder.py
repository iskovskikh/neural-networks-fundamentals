import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        super().__init__()
        ## YOUR CODE HERE

        assert len(channels) >= 2

        input_ = channels[0]
        hidden_ = channels[1:-1]
        output_ = channels[-1]

        print(f'{input_=}')
        print(f'{hidden_=}')
        print(f'{output_=}')
        
        # --- encoder ---
        in_features = input_
        encoder_layers = []

        for out_features in hidden_:
            encoder_layers.append(torch.nn.Linear(in_features, out_features))
            encoder_layers.append(activation())
            in_features = out_features

        encoder_layers.append(torch.nn.Linear(in_features, output_))

        self.encoder = torch.nn.Sequential(*encoder_layers)

        # --- decoder ---
        in_features = output_
        decoder_layers = []

        for out_features in reversed(hidden_):
            decoder_layers.append(torch.nn.Linear(in_features, out_features))
            decoder_layers.append(activation())
            in_features = out_features

        decoder_layers.append(torch.nn.Linear(in_features, input_))

        self.decoder = torch.nn.Sequential(*decoder_layers)

        if not hasattr(self, 'encoder'):
            self.encoder = torch.nn.Identity()
        if not hasattr(self, 'decoder'):
            self.decoder = torch.nn.Identity()

    def __forward_kernel(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE

        res = signal.reshape(signal.shape[0], -1)
        res = self.encoder(res)
        res = self.decoder(res)

        res = res.reshape(input_shape)
        return res

    def forward(self, batch):
        ## YOUR CODE HERE

        reconstruction = self.__forward_kernel(batch['data']['image'])

        if 'signals' not in batch:
            # batch['signals'] = {'reconstruction': batch['data']['image']}
            batch['signals'] = {'reconstruction': reconstruction}
        return batch
