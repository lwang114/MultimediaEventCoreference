import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from itertools import combinations
from models import SimplePairWiseClassifier
import json
  
class TripletLoss(nn.Module):
  def __init__(self, config):
    super(TripletLoss, self).__init__()
    self.simtype = config.get('simtype', 'mean_max')

  def forward(self, text_outputs, image_outputs,        
              text_mask, image_mask,
              margin=1.):
    '''
    :param text_outputs: FloatTensor of size (batch size, max num. of tokens, text embed dim)
    :param image_outputs: FloatTensor of size (batch size, max num. of image regions, image embed dim)
    :return scores: FloatTensor of size (batch size, max num. of [score_type] spans, span embed dim)
    '''
    B, N, _ = first.size()
    L = second.size(1)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    ntokens = text_mask.sum(-1)
    nregions = image_mask.sum(-1)

    if B == 1:
      return loss
    for i in range(B):
      I_imp_ind = i
      S_imp_ind = i
      while I_imp_ind == i:
        I_imp_ind = np.random.randint(0, n)
      while S_imp_ind == i:
        S_imp_ind = np.random.randint(0, n)
      
      nT = ntokens[i]
      nTimp = ntokens[S_imp_ind]

      if len(nregions):
        nR = nregions[i]
        nRimp = nregions[I_imp_ind]
    
      anchorsim = self.matchmap_similarity(
          self.compute_matchmap(text_outputs[i, :nT[i]], 
                                image_outputs[i, :nR[i]])
      )
      Simpsim = self.matchmap_similarity(
          self.compute_matchmap(text_outputs[S_imp_ind, :nT[S_imp_ind]],
                                image_outputs[i, :nR[i]])
      )
      Iimpsim = self.matchmap_similarity(
          self.compute_matchmap(text_outputs[i, :nT[i]],
                                image_outputs[I_imp_ind, :nR[I_imp_ind]])
      )
      
      S2I_simdif = margin + Iimpsim - anchorsim
      if (S2I_simdif.data > 0).all():
        loss = loss + S2I_simdif
      I2S_simdif = margin + Simpsim - anchorsim
      if (I2S_simdif.data > 0).all():
        loss = loss + I2S_simdif
    loss = loss / B
    return loss
    
  def compute_matchmap(self, S, I):
    return torch.mm(S, I.t())

  def matchmap_similarity(self, M, simtype):
    if simtype == 'mean':
      return M.mean()
    elif simtype == 'mean_max':
      return M.max(1)[0].mean()
    elif simtype == 'max_mean':
      return M.max(0)[0].mean()

  def retrieve(self, text_outputs, image_outputs, text_masks, image_masks):
    n = len(text_outputs)
    nF = text_masks.sum(-1)
    nR = image_masks.sum(-1) 

    S = torch.zeros((n, n), dtype=torch.float, requires_grad=False)
    for s_idx in range(n):
      for v_idx in range(n):
        S[s_idx, v_idx] = -self.matchmap_similarity(
                            self.compute_matchmap(text_outputs[s_idx][:nF[s_idx]],
                                                  image_outputs[v_idx][:nR[v_idx]])
                            ).to(device)

    _, I2S_idxs = S.topk(k, 0)
    _, S2I_idxs = S.topk(k, 1)
    return I2S_idxs.t(), S2I_idxs 
