import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from itertools import combinations
from models import SimplePairWiseClassifier
import json
  
class PairwiseGrounder(nn.Module):
  def __init__(self, config):
    super(PairwiseGrounder, self).__init__()
    # self.grounding_scorer = BilinearPairWiseClassifier(config)
    self.text_only_decode = config.get('text_only_decode', False)

  def score_grounding(self, first, second,
                      first_mask, second_mask, 
                      score_type='both'):
    '''
    :param first: FloatTensor of size (batch size, max num. of first spans, span embed dim)
    :param second: FloatTensor of size (batch size, max num. of second spans, span embed dim)
    :param score_type: str from {'first', 'both'}
    :return scores: FloatTensor of size (batch size, max num. of [score_type] spans, span embed dim)
    '''
    B = first.size(0)
    N = first.size(1)
    L = second.size(1)

    # Flatten the first and second into (batch size * max span num, span embed dim)
    sent_scores = torch.zeros((B, B), dtype=torch.float, device=first.device)
    token_scores = torch.zeros((B, N, L), dtype=torch.float, device=first.device)
    for idx in range(B):
      # Compute index pairs for all combinations between first and second
      # first_idxs = [i for i in range(N) for j in range(L)] 
      # second_idxs = [j for i in range(N) for j in range(L)] 
      # first_idxs = torch.LongTensor(first_idxs)
      # second_idxs = torch.LongTensor(second_idxs)

      # Compute sentence-level scores
      for jdx in range(B):
        mask = first_mask[idx].unsqueeze(-1) * second_mask[jdx].unsqueeze(0)
        avg_mask = torch.where(mask != 0, mask, torch.tensor(-9e9, device=first.device))
        # avg_mask_first = F.softmax(avg_mask, dim=0)
        # avg_mask_second = F.softmax(avg_mask, dim=1)
        att_weights = torch.matmul(first[idx], second[jdx].t()) * mask
        att_weights = torch.where(att_weights * mask != 0, att_weights, torch.tensor(-9e9, device=first.device))
        att_weights_first = F.softmax(att_weights, dim=-1) * mask
        score = (att_weights_first * att_weights).sum(dim=-1).mean() # * avg_mask_first).sum()

        att_weights_second = F.softmax(att_weights.t(), dim=-1) * mask.t()
        score = score + (att_weights_second.t() * att_weights).sum(dim=-1).mean() # * avg_mask_second).sum()
        sent_scores[idx, jdx] = score
        if idx == jdx:
          token_scores[idx] = att_weights

    return sent_scores, token_scores
    
  def calculate_loss(self, text_emb, 
                     image_emb,  
                     text_mask, 
                     image_mask):
    B = text_emb.size(0) 
    D = text_emb.size(-1)
    # Compute visual grounding scores
    S, att_weights = self.score_grounding(text_emb, image_emb, text_mask, image_mask)
    text_emb_i2s = F.softmax(att_weights, dim=-1) @ image_emb
    return S, text_emb_i2s 
  
  def forward(self, text_embeddings, 
              image_embeddings, 
              text_mask, 
              image_mask):
    '''
    :param span_embeddings: FloatTensor of size (batch size, max num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (batch size, max num. of ROIs, image embed dim), 
    :param entity_mappings: LongTensor of size (batch size, max num. of entities, num. of head entity mentions for coreference),
    :param event_mappings: LongTensor of size (batch size, max num. of head event mentions for coreference),
    :return score: FloatTensor of size (batch size,),
    '''
    n = text_embeddings.size(0)
    m = nn.LogSoftmax(dim=1)
    S, text_embeddings_i2s = self.calculate_loss(
                                 text_embeddings,
                                 image_embeddings,
                                 text_mask,
                                 image_mask)
    loss = -torch.sum(m(S).diag())-torch.sum(m(S.transpose(0, 1)).diag()) 
    loss = loss / n
    return loss, text_embeddings_i2s
    
  def retrieve(self, span_embeddings, image_embeddings, span_masks, image_masks, k=10):
    '''
    :param span_embeddings: FloatTensor of size (num. of docs, num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (num. of docs, num. of ROIs, image embed dim),
    '''
    n = len(span_embeddings)
    S = torch.zeros((n, n), dtype=torch.float, device=torch.device('cpu'))
    for s_idx in range(n):
      for v_idx in range(n):
        span_embedding = span_embeddings[s_idx].cuda()
        image_embedding = image_embeddings[v_idx].cuda()
        span_mask = span_masks[s_idx].cuda()
        image_mask = image_masks[v_idx].cuda()
        scores, _ = self.score_grounding(span_embedding,
                                         image_embedding,
                                         span_mask,
                                         image_mask)
        S[s_idx, v_idx] = scores.cpu()

    _, I2S_idxs = S.topk(k, 0)
    _, S2I_idxs = S.topk(k, 1) 
    return I2S_idxs.t(), S2I_idxs
