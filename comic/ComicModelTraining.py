import time

import torch
import torch.nn as nn
import torchinfo
from torch.utils.data import DataLoader
from torch.optim import Adam
from ComicDataset import ComicDataset
from video.VideoCompressionModel import VideoCompressionModel, get_model_dims
from ComicCompressionModel import CombinedLoss
from pathlib import Path
from itertools import chain
from typing_extensions import TextIO
from typing import Optional, Tuple, Any

type TrainingTools = Tuple[VideoCompressionModel, CombinedLoss, Adam, DataLoader[Any], DataLoader[Any]]

def comic_preparations() -> TrainingTools:
    p = Path.home() / "global" / "warehouse" / "comics"
    train = p / ".train"
    test = p / ".test"
    validation = p / ".validation"

    model = VideoCompressionModel(3, 16, 1, (1650, 2499), layers=5).to('cuda')

    loss_function = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters())

    training_path = train.rglob("*.epub")
    validation_path = validation.rglob("*.epub")

    training_dataset = ComicDataset(training_path)
    validation_dataset = ComicDataset(validation_path)

    training_loader = DataLoader(training_dataset, batch_size=2, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=2, shuffle=True)

    return model, loss_function, optimizer, training_loader, validation_loader

def epoch_step(training: TrainingTools, num_epochs: int = 10, log_file: Optional[TextIO] = None) -> None:
    model, loss_function, optimizer, training_loader, validation_loader = training

    start = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} - delta: {time.time() - start}")
        start = time.time()
        model.train()
        running_loss = 0.0

        for tensor in training_loader:
            tensor = tensor.cuda()

            optimizer.zero_grad()

            reconstructed = model(tensor)

            loss = loss_function(reconstructed, tensor)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(training_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}", sep=" - ", end="")
        log_file.write(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}\n")

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for tensor in validation_loader:
                tensor = tensor.cuda()
                reconstructed = model(tensor)
                validation_loss += loss_function(reconstructed, tensor).item()

        avg_val_loss = validation_loss / len(validation_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        log_file.write(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}\n")


if __name__ == "__main__":
    with open("train.log", "w") as log:
        epoch_step(comic_preparations(), log_file=log)