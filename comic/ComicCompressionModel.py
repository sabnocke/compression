import torch
import torch.nn as nn

from video.VideoCompressionModel import get_model_dims, VideoCompressionModel
from pytorch_msssim import SSIM, ssim

class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.85):
        super(CombinedLoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, x, y):
        mse = self.mse_loss(x, y)

        _ssim = ssim(x, y, data_range=1.0, size_average=True)
        ssim_loss = 1 - _ssim

        combined = self.alpha * mse + (1 - self.alpha) * ssim_loss
        return combined

if __name__ == "__main__":
    model = VideoCompressionModel(3, 32, 1,
                                  io_dimensions=get_model_dims(1625, 2498))