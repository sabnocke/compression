import torch
from torch.utils.data import DataLoader, Dataset
import ffmpeg
import av
import numpy as np
from pathlib import Path
from typing import Iterable, Sized

class VideoDataset(Dataset):
    def __init__(self, source: Iterable[Path], num_frames_per_chunk: int = 16):
        self.source = tuple(source)
        self.num_frames_per_chunk = num_frames_per_chunk

        self.total_chunks = []
        for path in source:
            container = av.open(path)
            total_frames = container.streams.video[0].frames
            container.close()

            for i in range(0, total_frames, self.num_frames_per_chunk):
                self.total_chunks.append((path, i))


    def __len__(self) -> int:
       return len(self.source)

    def __getitem__(self, index: int):
        curr_path, start = self.total_chunks[index]
        frames = []
        container = av.open(curr_path)
        container.seek(start, stream=container.streams.video[0])

        for i, frame in enumerate(container.decode(video=0)):
            if i >= self.num_frames_per_chunk:
                break

            np_array = frame.to_rgb().to_ndarray()

            # Perform any preprocessing here (e.g., resizing) on the NumPy array

            tensor_frame = torch.from_numpy(np_array).permute(2, 0, 1)
            frames.append(tensor_frame)


        container.close()
        return torch.stack(frames)