import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from models import SimplePairWiseClassifier  

class ResNet101(nn.Module):
  def __init__(self, embedding_dim=1024, device=torch.device('cpu')):
    super(ResNet101, self).__init__()
    net = getattr(models, 'resnet101')(pretrained=True)
    b = list(net.children())
    self.backbone = nn.Sequential(*b[:-2])
    self.pooler = nn.Sequential(*[b[-2]])
    self.embedder = nn.Linear(2048, embedding_dim)
    
    for p in self.backbone.parameters():
      p.requires_grad = False
    for p in self.pooler.parameters():
      p.requires_grad = False
     
    self.device = device
    self.to(device)
     
  def forward(self, images, mask=None, return_feat=False):    
    images = images.to(self.device)
    if mask is not None:
      mask = mask.to(self.device)
    
    ndim = images.ndim
    B, L = images.size(0), images.size(1)
    if ndim == 5:
      H, W, C = images.size(2), images.size(3), images.size(4)
      images = images.view(B*L, H, W, C)

    fmap = self.backbone(images)
    fmap = self.pooler(fmap)
    emb = self.embedder(fmap.permute(0, 2, 3, 1))

    _, He, We, D = emb.size()
    if ndim == 5:
      emb = emb.view(B, L*He*We, D)
      fmap = fmap.view(B, L, -1, He, We)
      if mask is not None:
        mask = mask.unsqueeze(-1).repeat(1, 1, He*We).flatten(start_dim=1)
    else:
      emb = emb.view(B, He*We, D)
    
    if return_feat:
      return emb, fmap, mask
    else: 
      return emb, mask

  
class ResNet152(nn.Module):
  def __init__(self, embedding_dim=1024, device=torch.device('cpu')):
    super(ResNet152, self).__init__()
    net = getattr(models, 'resnet152')(pretrained=True)
    b = list(net.children())
    self.backbone = nn.Sequential(*b[:-2])
    self.pooler = nn.Sequential(*[b[-2]])
    self.embedder = nn.Linear(2048, embedding_dim)
    
    for p in self.backbone.parameters():
      p.requires_grad = False
    for p in self.pooler.parameters():
      p.requires_grad = False
     
    self.device = device
    self.to(device)
     
  def forward(self, images, mask=None, return_feat=False):    
    images = images.to(self.device)
    if mask is not None:
      mask = mask.to(self.device)
    
    ndim = images.ndim
    B, L = images.size(0), images.size(1)
    if ndim == 5:
      H, W, C = images.size(2), images.size(3), images.size(4)
      images = images.view(B*L, H, W, C)

    fmap = self.backbone(images)
    fmap = self.pooler(fmap)
    emb = self.embedder(fmap.permute(0, 2, 3, 1))

    _, He, We, D = emb.size()
    if ndim == 5:
      emb = emb.view(B, L*He*We, D)
      fmap = fmap.view(B, L, -1, He, We)
      if mask is not None:
        mask = mask.unsqueeze(-1).repeat(1, 1, He*We).flatten(start_dim=1)
    else:
      emb = emb.view(B, He*We, D)
    
    if return_feat:
      return emb, fmap, mask
    else: 
      return emb, mask

class NoOp(nn.Module):
  def __init__(self):
    super(NoOp, self).__init__()

  def forward(self, x, mask=None, return_feat=False):
    if return_feat:
      return x, x, mask
    else:
      return x, mask
