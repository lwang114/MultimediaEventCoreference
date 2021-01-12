import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

class NoOp(nn.Module):
  def __init__(self):
    super(NoOp, self).__init__()

  def forward(self, x):
    return x

class BiLSTM(nn.Module):
  def __init__(self, input_dim, embedding_dim, num_layers=1, char_dim=0, char_vec=None):
    super(BiLSTM, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_layers = num_layers
    self.batchnorm1 = nn.BatchNorm2d(1) 
    self.cnn = nn.Conv2d(1, input_dim, kernel_size=(input_dim, 3), stride=(1,1), padding=(0,1))
    self.rnn = nn.LSTM(input_size=input_dim+char_dim, hidden_size=embedding_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
    if not char_vec is None:
      self.batchnorm2 = nn.BatchNorm2d(char_vec.shape[1])
      self.char_emb = nn.Embedding(char_vec.shape[0], char_vec.shape[1])
      self.char_emb.weight.data.copy_(torch.from_numpy(char_vec))
      self.char_cnn = nn.Conv2d(char_vec.shape[1], char_dim, kernel_size=(3, 1), stride=(1,1), padding=(0,0))


  def forward(self, x, save_features=False, char_idxs=None):
    if x.dim() < 3:
      x = x.unsqueeze(0)
    B = x.size(0)
    T = x.size(1)
    
    x = x.unsqueeze(1)
    x = torch.transpose(x, -2, -1)
    x = self.batchnorm1(x)
    x = F.relu(self.cnn(x)).squeeze(2).transpose(-2, -1)
    if not char_idxs is None:
      x_char = self.char_emb(char_idxs).permute(0, 3, 2, 1)
      x_char = self.batchnorm2(x_char)
      x_char = F.relu(self.char_cnn(x_char)).max(dim=-2)[0]
      x_char = torch.transpose(x_char, -2, -1)
      x = torch.cat([x, x_char], dim=-1)
    h0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
    c0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
    if torch.cuda.is_available():
      h0 = h0.cuda()
      c0 = c0.cuda()
    embed, _ = self.rnn(x, (h0, c0))
    outputs = []
    for b in range(B):
      outputs.append(embed[b])
    outputs = torch.stack(outputs, dim=1).transpose(0, 1)
    return outputs

class CharCNN(nn.Module):
  def __init__(self, input_dim, embedding_dim, char_vec):
    super(CharCNN, self).__init__()
    self.char_dim = char_vec.shape[1]
    self.embedding_dim = embedding_dim

    # Initialize the character embeddings
    self.char_emb = nn.Embedding(char_vec.shape[0], char_vec.shape[1])
    self.char_emb.weight.data.copy_(torch.from_numpy(char_vec))

    self.batchnorm1 = nn.BatchNorm2d(self.char_dim)
    self.conv1 = nn.Conv2d(self.char_dim, 100, kernel_size=(input_dim, 3), stride=(1,1), padding=(0,0))
    self.conv2 = nn.Conv2d(100, 200, kernel_size=(1,3), stride=(1,1), padding=(0,1))
    self.conv3 = nn.Conv2d(200, embedding_dim, kernel_size=(1,3), stride=(1,1), padding=(0,2))
    
  def forward(self, x):
    if x.ndim < 3:
      x = x.unsqueeze(0)
    
    # Shape: batch size x num. of tokens x num. of chars x embedding dim 
    x = self.char_emb(x)

    # Shape: batch size x embedding dim x num. of chars x num. of tokens 
    x = x.permute(0, 3, 2, 1)
    
    x = self.batchnorm1(x)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.squeeze(2)
    x = torch.transpose(x, -2, -1)
    return x 
