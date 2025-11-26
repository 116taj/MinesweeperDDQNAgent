import torch
import torch.nn as nn
#CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 4 conv layers with 128 filters and relu as activation
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        #flatten
        x = x.view(x.size(0), -1)
        return x
