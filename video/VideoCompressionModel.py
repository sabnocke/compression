from typing import Tuple

import torch
import torch.nn as nn
from functools import reduce
from operator import mul

import warnings
from pprint import pprint


def get_model_dims(width: int, height: int, layers: int = 3, init_channels: int = 3, /, *, noprint: bool = True) -> tuple[int, tuple[int, ...]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")


        layers_gen = []
        for j in range(layers):
            layers_gen.append(
                nn.Conv2d(init_channels if j == 0 else 64 << (j - 1), 64 << j, kernel_size=3, padding=1)
            )
            layers_gen.append(nn.ReLU())
            layers_gen.append(nn.MaxPool2d(kernel_size=2, stride=2))

        dummy_encoder = nn.Sequential(
            *layers_gen,
        )

        dummy_input = torch.randn(1, init_channels, height, width)
        decoder_input_shape: tuple[int, ...] = tuple(dummy_encoder(dummy_input).shape[1:])
        encoder_output_size: int = reduce(mul, decoder_input_shape)
        if not noprint:
            pprint(f"[{width}x{height}, layers: {layers}] "
                   f"decoder_output_shape: {decoder_input_shape}, "
                   f"GRU input size: {reduce(mul, decoder_input_shape):,}")

        return encoder_output_size, decoder_input_shape


class VideoCompressionModel(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int,
                 /,
                 size: Tuple[int, int],
                 *,
                 layers: int = 3 ):
        super(VideoCompressionModel, self).__init__()

        self.in_channels: int = in_channels
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.encoder_output_size, self.decoder_input_shape = get_model_dims(size[0], size[1], layers, in_channels, noprint=False)
        self.layers: int = layers

        # 1. Encoder
        # This CNN reduces a frame to a compressed vector.
        self.encoder = nn.Sequential(
            *self.generate_encoder()
        )

        # 2. The GRU Layer
        # This processes the sequence of vectors from the encoder.
        print(self.encoder_output_size)
        self.gru = nn.GRU(input_size=self.encoder_output_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # 3. Decoder
        # This reconstructs the frames from the GRU's output.
        self.decoder_input_layer = nn.Sequential(
            nn.Linear(hidden_dim, self.encoder_output_size),   # A linear layer to map hidden_dim back to flattened encoder size
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            *self.generate_decoder()
        )

    def generate_encoder(self):
        generated_layers = []
        for layer in range(self.layers):
            print(layer, self.in_channels if layer == 0 else 64 << (layer - 1), 64 << layer)

            generated_layers.append(
                nn.Conv2d(self.in_channels if layer == 0 else 64 << (layer - 1), 64 << layer, kernel_size=3, padding=1)
            )
            generated_layers.append(nn.ReLU())
            generated_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return generated_layers

    def generate_decoder(self):
        generated_layers = []
        f_a = lambda x: -x + 2
        f_b = lambda channels, x: channels if x == 0 else 64 << (x - 1)
        for layer in range(self.layers -1, -1, -1):
            print(layer, 64 << layer,  f_b(self.in_channels, layer))
            generated_layers.append(
                nn.ConvTranspose2d(64 << layer, f_b(self.in_channels, layer), kernel_size=4, stride=2, padding=1)
            )

            if layer < self.layers - 1:
                generated_layers.append(nn.ReLU())

        generated_layers.append(
            nn.Sigmoid()
        )

        return generated_layers

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()
        print(h, w)

        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        reconstructed_frames = []
        for i in range(num_frames):
            frame = x[:, i, :, :, :]

            # encode the current frame
            encoded_frame = self.encoder(frame)

            # reshape frame for the GRU and pass to the GRU with previous hidden state
            gru_input = encoded_frame.flatten(start_dim=1).unsqueeze(1)
            gru_output, hidden_state = self.gru(gru_input, hidden_state)

            decoded_frame_flat = self.decoder_input_layer(gru_output.squeeze(1))
            print(decoded_frame_flat.size())
            reconstructed_frames.append(self.decoder(decoded_frame_flat.view(batch_size, *self.decoder_input_shape)))

        return torch.stack(reconstructed_frames, dim=1)