import torch
from torch import nn
import torch.nn.functional as F

class UNetFull(nn.Module):
  def __init__(self, num_classes,  input_filters = 3, start_features= 64):
    super().__init__()

    self.num_classes = num_classes
    self.input_filters = input_filters
    self.start_features = start_features

    self.down_layers1 = nn.Sequential(
      nn.Conv2d(self.input_filters, self.start_features, 3, padding = 1),
      nn.BatchNorm2d(self.start_features),
      nn.ReLU(),
      nn.Conv2d(self.start_features, self.start_features, 3, padding = 1),
      nn.BatchNorm2d(self.start_features),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )

    self.down_layers2 = nn.Sequential(
      nn.Conv2d(self.start_features, self.start_features * 2, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 2),
      nn.ReLU(),
      nn.Conv2d(self.start_features * 2, self.start_features * 2, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 2),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )

    self.down_layers3 = nn.Sequential(
      nn.Conv2d(self.start_features * 2, self.start_features * 4, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 4),
      nn.ReLU(),
      nn.Conv2d(self.start_features * 4, self.start_features * 4, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 4),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.down_layers4 = nn.Sequential(
      nn.Conv2d(self.start_features * 4, self.start_features * 8, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 8),
      nn.ReLU(),
      nn.Conv2d(self.start_features * 8, self.start_features * 8, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 8),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.base_layer = nn.Sequential(
      nn.Conv2d(self.start_features * 8, self.start_features * 16, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 16),
      nn.ReLU(),
      nn.Conv2d(self.start_features * 16, self.start_features * 8, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 8),
      nn.ReLU()
    )
    self.up_layers1 = nn.Sequential(
      nn.Conv2d(self.start_features * 16, self.start_features * 16, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 16),
      nn.ReLU(),
      nn.Conv2d(self.start_features * 16, self.start_features * 8, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 8),
      nn.ReLU()
    )
    self.up_layers2 = nn.Sequential(
      nn.Conv2d(self.start_features * 12, self.start_features * 8, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 8),
      nn.ReLU(),
      nn.Conv2d(self.start_features * 8, self.start_features * 4, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 4),
      nn.ReLU()
    )
    self.up_layers3 = nn.Sequential(
      nn.Conv2d(self.start_features * 6, self.start_features * 4, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 4),
      nn.ReLU(),
      nn.Conv2d(self.start_features * 4, self.start_features * 2, 3, padding = 1),
      nn.BatchNorm2d(self.start_features * 2),
      nn.ReLU()
    )
    self.up_layers4 = nn.Sequential(
      nn.Conv2d(self.start_features * 3, self.start_features, 3, padding = 1),
      nn.BatchNorm2d(self.start_features),
      nn.ReLU(),
      nn.Conv2d(self.start_features, self.start_features, 3, padding = 1),
      nn.BatchNorm2d(self.start_features),
      nn.ReLU(),
      nn.Conv2d(self.start_features, num_classes, 1)
    )

    self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)

  def forward(self, x):

    down1 = self.down_layers1(x)
    down2 = self.down_layers2(down1)
    down3 = self.down_layers3(down2)
    down4 = self.down_layers4(down3)
    x = self.base_layer(down4)  
    x = self.upsample_and_concat(x, down4)
    y = self.up_layers1(x)
    y = self.upsample_and_concat(y, down3)
    y = self.up_layers2(y)
    y = self.upsample_and_concat(y, down2)
    y = self.up_layers3(y)
    y = self.upsample_and_concat(y, down1)
    y = self.up_layers4(y)
    return y

  def upsample_and_concat (self, x, down):
    
    x = self.up(x)
    diffY = torch.tensor([x.size()[2] - down.size()[2]])
    diffX = torch.tensor([x.size()[3] - down.size()[3]])
    
    down = F.pad(down, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    
    x = torch.cat([down, x], dim = 1)
    return x
        