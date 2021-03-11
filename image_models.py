import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualEncoder(nn.Module):
  def __init__(self, input_dim, embedding_dim, num_layers=1):
    super(VisualEncoder, self).__init__()
    self.conv = nn.Conv2d(1, embedding_dim, kernel_size=(400, 5), padding=(0,2))
    self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
    self.encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=1)

  def forward(self, x):
    if x.ndim == 3:
      x = x.permute(0, 2, 1).unsqueeze(1)
    x = F.relu(self.conv(x))
    x = self.pool(x).squeeze(2).permute(0, 2, 1) 
    return self.encoder(x)
