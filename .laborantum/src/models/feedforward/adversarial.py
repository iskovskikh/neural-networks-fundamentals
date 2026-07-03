import torch
import copy


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, strength):
        ctx.strength = strength
        return signal.view_as(signal)

    @staticmethod
    def backward(ctx, grad_output):
        ### YOUR CODE HERE

        return -ctx.strength * grad_output, None

        # return grad_output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = float(strength)

    def forward(self, signal):
        return GradientReversalFunction.apply(signal, self.strength)


class GAN(torch.nn.Module):
    def __init__(
            self,
            channels,
            gradient_reversal_strength=1.0,
            activation=lambda: torch.nn.LeakyReLU(negative_slope=0.5)
        ):
        ...
        ## YOUR CODE HERE

        super().__init__()

        noise_dim = channels[0]
        hidden_ = channels[1:-1]
        output_ = channels[-1]

        # In `__init__`, create these modules:
        # 
        # - `self.generator_discriminator_bridge = GradientReversalLayer(gradient_reversal_strength)`;
        self.generator_discriminator_bridge = GradientReversalLayer(gradient_reversal_strength)

        # - `self.gradient_reversal` as an alias of the same layer;
        self.gradient_reversal = self.generator_discriminator_bridge

        # - `self.generator`: fully connected layers following `channels` from left to right.
        #  For this task the check uses `[noise_dim, 512, 28 * 28]`,
        #  so the generator path is `noise_dim -> 512 -> 784`.
        #  Use a fresh copy of the activation after each hidden linear layer,
        #  then `torch.nn.Tanh()` at the end so generated pixels live near the MNIST `[-1, 1]` scale;

        in_features = noise_dim
        generator_layers = []

        for out_features in hidden_:
            generator_layers.append(torch.nn.Linear(in_features, out_features))
            generator_layers.append(copy.deepcopy(activation()))
            in_features = out_features

        generator_layers.append(torch.nn.Linear(in_features, output_))
        generator_layers.append(torch.nn.Tanh())

        self.generator = torch.nn.Sequential(*generator_layers)

        # - `self.discriminator`: fully connected layers following `channels` in reverse.
        #  For `[noise_dim, 512, 28 * 28]`, the discriminator path is `784 -> 512 -> noise_dim`,
        #  again using fresh activation modules between linear layers;
        in_features = output_
        discriminator_layers = []

        for out_features in reversed(hidden_):
            discriminator_layers.append(torch.nn.Linear(in_features, out_features))
            discriminator_layers.append(copy.deepcopy(activation()))
            in_features = out_features

        discriminator_layers.append(torch.nn.Linear(in_features, noise_dim))

        self.discriminator = torch.nn.Sequential(*discriminator_layers)

        # - `self.classifier`: a final `Linear(noise_dim, 1)` that produces
        #  one raw real/fake logit per image.
        #
        self.classifier = torch.nn.Linear(noise_dim, 1)

        # The default activation is `LeakyReLU(negative_slope=0.5)`.
        #  Keep that high negative slope unless you are deliberately experimenting:
        #  it helps fight low-gradient behavior when the discriminator gets too confident too early.

    def discriminate(self, signal):
        signal = signal.reshape(signal.shape[0], -1)
        features = self.discriminator(signal)
        return self.classifier(features).flatten()

    def forward(self, batch):
        ## YOUR CODE HERE

        # In `forward`, implement this data path:

        # `noise 
        # -> generator 
        # -> gradient reversal 
        # -> concatenate with real images 
        # -> discriminator 
        # -> classifier`

        # The model reads `batch['data']['noise']` 
        # and either `batch['data']['real']`
        # or `batch['data']['image']`.

        noise = batch['data'].get('noise')
        real = batch['data'].get('real')
        if real is None:
            real = batch['data'].get('image')


        # Flatten real images before concatenation.
        # The classifier outputs raw logits:
        # larger logits mean "more real",
        # smaller logits mean "more fake".

        # `forward` should mutate and return the same batch dictionary.
        # Keep the original `batch['data']` inputs in place,
        # and add two new dictionaries:
        # `batch['signals']` for tensors used by losses/metrics
        # and `batch['postprocessed']` for easy inspection.

        # If the noise batch has size `B`,
        # then `generated` has shape `(B, 784)`.
        # When real images are present, run the discriminator on fake samples first and real samples second,
        # so the combined discriminator logits have shape `(2 * B,)`,
        # `fake_logits` are the first `B` entries, and `real_logits` are the final `B` entries.

        generated = self.generator(noise)

        

        fake = self.gradient_reversal(generated)

        if real is not None:
            real = real.reshape(real.shape[0], -1)

            discriminator_input = torch.cat([fake, real], dim=0)

            discriminator_logits = self.discriminate(discriminator_input)

            B = fake.shape[0]
            fake_logits = discriminator_logits[:B]
            real_logits = discriminator_logits[B:]

            batch['signals'] = dict(
                generated= generated,
                discriminator_logits = discriminator_logits,
                fake_logits = fake_logits,
                real_logits = real_logits,
                discriminator_scores = discriminator_logits,
                fake_scores = fake_logits,
                real_scores = real_logits,

            )
            batch['postprocessed'] = dict(
                discriminator_score = discriminator_logits,
                fake_score = fake_logits,
                real_score = real_logits,
                discriminator_probability = torch.sigmoid(fake_logits),
                fake_probability = torch.sigmoid(fake_logits),
                real_probability = torch.sigmoid(real_logits),
            )

        else:
            fake_logits = self.discriminate(fake)
            real_logits = fake_logits

            batch['signals'] = dict(
                generated= generated,
                discriminator_logits = discriminator_logits,
                fake_logits = fake_logits,
                # real_logits = real_logits,
                discriminator_scores = discriminator_logits,
                fake_scores = fake_logits,
                # real_scores = real_logits,

            )
            batch['postprocessed'] = dict(
                discriminator_score = discriminator_logits,
                fake_score = fake_logits,
                # real_score = real_logits,
                discriminator_probability = torch.sigmoid(fake_logits),
                fake_probability = torch.sigmoid(fake_logits),
                # real_probability = torch.sigmoid(real_logits),
            )
        

        # Use exactly these key names. The required outputs are:

        # - `batch['signals']['generated']`;
        # - `batch['signals']['discriminator_logits']` for the concatenated fake plus real batch;
        # - `batch['signals']['fake_logits']`;
        # - `batch['signals']['real_logits']` when real images are present;
        # - matching convenience aliases `discriminator_scores`, `fake_scores`, and `real_scores`;
       
        # - `batch['postprocessed']['discriminator_score']`,
        # `batch['postprocessed']['fake_score']`, and
        # `batch['postprocessed']['real_score']` when real images are present;
        # - `batch['postprocessed']['discriminator_probability']`,
        # `batch['postprocessed']['fake_probability']`, and
        # `batch['postprocessed']['real_probability']` when real images are present, computed with `torch.sigmoid`.
        


        # A separate `discriminate` helper is not part of the required student-facing API.
        # It may be kept as a small internal helper if the surrounding file already has it,
        # but the required behavior is the `forward` protocol above.

        if 'signals' not in batch:
            generated = batch['data'].get('noise')
            if generated is None:
                generated = torch.empty(0)
            batch['signals'] = {
                'generated': generated,
                'fake_scores': torch.zeros(generated.shape[0], device=generated.device),
                'fake_logits': torch.zeros(generated.shape[0], device=generated.device),
            }
            batch['postprocessed'] = {
                'fake_score': torch.zeros(generated.shape[0], device=generated.device),
                'fake_probability': torch.zeros(generated.shape[0], device=generated.device),
            }
        return batch
