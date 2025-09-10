from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn
from pathlib import Path
from collections.abc import Iterator
from PIL import Image
from preprocessor import epub_image_generator, find_max, resize_and_pad, parallel_resize_pad
from zipfile import ZipFile
from icecream import ic

class ComicDataset(Dataset):
    def __init__(self, images: Iterator[Path], num_chunks: int = 16):
        self.image_paths: List[Path] = []
        self.num_chunks: int = num_chunks
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])



        for image in images:
            self.image_paths.extend(epub_image_generator(image))

        max_w, max_h = find_max(self.image_paths)

        parallel_resize_pad(self.image_paths, (max_w, max_h))

        self.chunk_starts = [i for i in range(0, len(self.image_paths), num_chunks)]

        ic(self.chunk_starts[-1], len(self.image_paths))

    def __len__(self):
        return len(self.chunk_starts)

    def __getitem__(self, index):
        start_idx = self.chunk_starts[index]
        chunk_paths = self.image_paths[start_idx:start_idx + self.num_chunks]

        chunk_tensor = []
        for chunk_path in chunk_paths:
            img = Image.open(chunk_path).convert('RGB')
            if self.transform:
                chunk_tensor.append(self.transform(img))


        return torch.stack(chunk_tensor)
