import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, in_dim=100):
        super(Generator, self).__init__()
        # N x 100
        self.hidden0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim, out_channels=1024, kernel_size=4),
            nn.BatchNorm2d(1024),
            nn.ReLU(True))
        # N x 4 x 4 x 1024
        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True))
        # N x 8 x 8 x 512
        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        # N x 16 x 16 x 256
        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        # N x 32 x 32 x 128
        self.hidden4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh())
        # N x 64 x 64 x 3

    def forward(self, x):
        output = self.hidden0(x)
        output = self.hidden1(output)
        output = self.hidden2(output)
        output = self.hidden3(output)
        output = self.hidden4(output)
        return output
