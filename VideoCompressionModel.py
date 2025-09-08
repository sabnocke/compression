import torch
import torch.nn as nn
from functools import reduce
from operator import mul

import warnings


def get_model_dims(width: int, height: int) -> tuple[int, tuple[int, ...]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dummy_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more layers as desired
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        dummy_input = torch.randn(1, 3, height, width)
        decoder_output_shape: tuple[int, ...] = tuple(dummy_encoder(dummy_input).shape[1:])
        encoder_output_size: int = reduce(mul, decoder_output_shape)
        print(f"decoder_output_shape: {decoder_output_shape}, GRU input size: {reduce(mul, decoder_output_shape):,}")
        return encoder_output_size, decoder_output_shape


class VideoCompressionModel(nn.Module):
    def __init__(self, in_channels: int, hidden_dim, num_layers, encoder_output_size, decoder_input_shape):
        super(VideoCompressionModel, self).__init__()

        self.in_channels: int = in_channels
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.decoder_input_shape: tuple[int, ...] = decoder_input_shape
        self.encoder_output_size: int = encoder_output_size

        # 1. Encoder
        # This CNN reduces a frame to a compressed vector.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more layers as desired
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 2. The GRU Layer
        # This processes the sequence of vectors from the encoder.
        self.gru = nn.GRU(input_size=self.encoder_output_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # 3. Decoder
        # This reconstructs the frames from the GRU's output.
        self.decoder_input_layer = nn.Sequential(
            nn.Linear(hidden_dim, self.encoder_output_size),   # A linear layer to map hidden_dim back to flattened encoder size
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()    # Use Sigmoid to output pixel values between 0 and 1
        )

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()

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
            reconstructed_frames.append(self.decoder(decoded_frame_flat.view(batch_size, *self.decoder_output_shape)))

        return torch.stack(reconstructed_frames, dim=1)