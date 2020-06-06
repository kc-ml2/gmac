import torch
import torch.nn as nn


class DeepMindEnc(nn.Module):
    def __init__(self, input_shape, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU()
        )
        sample = torch.FloatTensor(1, *input_shape)
        sample = self.feature(sample)
        sample = sample.view(sample.size(0), -1)
        self.fc = nn.Sequential(
            nn.Linear(sample.size(-1), hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x/255.0
        x = self.feature(x)
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        return out
