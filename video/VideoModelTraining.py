import time

import torch
import torch.nn as nn
import torchinfo
from torch.utils.data import DataLoader
from VideoDataset import VideoDataset
from VideoCompressionModel import VideoCompressionModel, get_model_dims
from pathlib import Path
from itertools import chain
from sys import exit

p = Path.home() / "Videos" / "Movies"
training_p = p / ".train"
validation_p = p / ".validation"
test_p = p / ".test"

get_model_dims(1920, 1080)


# in_channels = 3 -> 3 channels of RGB
# hidden_dims = 256 -> capacity of GRU layer (?) [2^n where n >= 8]
# num_layers = 1 -> determines how many GRU layers are stacked [1, 2, 3 , 4, ... (?)]
model = VideoCompressionModel(3, 256, 1, *get_model_dims(1920, 1080))

torchinfo.summary(model, input_size=(8, 16, 3, 1080, 1920), depth=3, device="cuda")

# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Prepare datasets
training_path_collection = chain(
    training_p.rglob("*.mp4"),
    training_p.rglob("*.mkv"),
    training_p.rglob("*.avi")
)

validation_path_collection = chain(
    validation_p.rglob("*.mp4"),
    validation_p.rglob("*.mkv"),
    validation_p.rglob("*.avi")
)

training_dataset = VideoDataset(training_path_collection)
validation_dataset = VideoDataset(validation_path_collection)

training_loader = DataLoader(training_dataset, batch_size=1, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# Training loop
num_epochs = 10

training_log = open("training.log", "a+", encoding="utf-8")

start = time.time()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs} - delta: {time.time() - start}")
    start = time.time()
    model.train()
    running_loss = 0.0
    for video_tensor in training_loader:
        # Move tensor into GPU memory
        video_tensor = video_tensor.cuda()

        # Zero the grads from previous step
        optimizer.zero_grad()

        # Forward pass; get the model's output
        reconstructed_video = model(video_tensor)

        # Calculate the loss
        loss = loss_function(reconstructed_video, video_tensor)

        # Backward pass; compute gradients
        loss.backward()

        # Update the model's weights
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(training_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}", sep=" - ", end="")

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for video_tensor in validation_loader:
            video_tensor = video_tensor.cuda()
            reconstructed_video = model(video_tensor)
            loss = loss_function(reconstructed_video, video_tensor)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(validation_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    training_log.write(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}\n")

