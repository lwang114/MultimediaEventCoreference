import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from itertools import combinations
from models import SimplePairWiseClassifier, BilinearPairWiseClassifier 
import json
  
class JointSupervisedGroundedCoreferencer(nn.Module):
  def __init__(self, config):
    super(JointSupervisedGroundedCoreferencer, self).__init__()
    self.text_scorer = SimplePairWiseClassifier(config) 
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
    mention_scores = torch.zeros((B, N, L), dtype=torch.float, device=first.device)
    sent_scores = torch.zeros((B, B), dtype=torch.float, device=first.device)
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
        avg_mask_first = F.softmax(avg_mask, dim=0)
        avg_mask_second = F.softmax(avg_mask, dim=1)

        att_weights = torch.matmul(first[idx], second[jdx].t()) * mask
        att_weights = torch.where(att_weights * mask != 0, att_weights, torch.tensor(-9e9, device=first.device))
        att_weights_first = F.softmax(att_weights, dim=-1) * mask
        score = (att_weights_first * att_weights * avg_mask_first).sum()

        att_weights_second = F.softmax(att_weights.t(), dim=-1) * mask.t()
        score = score + (att_weights_second.t() * att_weights * avg_mask_second).sum()
        sent_scores[idx, jdx] = score

        if idx == jdx: # Compute mention-level scores
          mention_scores[idx] = att_weights

    return sent_scores, mention_scores
  
  def score_text(self, first, second):
    B = first.size(0)
    N = first.size(1)
    scores = self.text_scorer(first, second)
    return scores
  
  def score_image(self, first, second):
    scores = torch.sum(first * second, dim=-1)
    scores = scores / (torch.norm(first, dim=-1) * torch.norm(second, dim=-1)) 
    return scores

  def calculate_loss(self, text_emb, span_emb, image_emb, text_mask, span_mask, image_mask):
    B = text_emb.size(0) 
    D = text_emb.size(-1)
    N = span_emb.size(1)
    # Compute visual grounding scores
    S, _ = self.score_grounding(text_emb, image_emb, text_mask, image_mask)
    _, grounding_scores = self.score_grounding(span_emb.view(B, N, -1, D).mean(dim=2), image_emb, span_mask, image_mask)
    first_idx, second_idx = zip(*list(combinations(range(N), 2)))
    text_scores = self.score_text(span_emb[first_idx].view(B*len(first_idx), -1), span_emb[second_idx].view(B*len(second_idx), -1)).view(B, len(first_idx))
    return S, grounding_scores, text_scores 
  
  def forward(self, text_embeddings, span_embeddings, image_embeddings, text_mask, span_mask, image_mask):
    '''
    :param span_embeddings: FloatTensor of size (batch size, max num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (batch size, max num. of ROIs, image embed dim), 
    :param entity_mappings: LongTensor of size (batch size, max num. of entities, num. of head entity mentions for coreference),
    :param event_mappings: LongTensor of size (batch size, max num. of head event mentions for coreference),
    :return score: FloatTensor of size (batch size,),
    '''
    self.text_scorer.train()
    n = span_embeddings.size(0)
    m = nn.LogSoftmax(dim=1)
    S, grounding_scores, text_scores = self.calculate_loss(
                                 text_embeddings,
                                 span_embeddings,
                                 image_embeddings,
                                 text_mask,
                                 span_mask,
                                 image_mask)
    loss = -torch.sum(m(S).diag())-torch.sum(m(S.transpose(0, 1)).diag())
    
    loss = loss / n
    return loss, grounding_scores, text_scores
    
  def retrieve(self, span_embeddings, image_embeddings, span_masks, image_masks, k=10):
    '''
    :param span_embeddings: FloatTensor of size (num. of docs, num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (num. of docs, num. of ROIs, image embed dim),
    '''
    self.text_scorer.eval()
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
    
  def predict(self, text_embeddings, span_embeddings, image_embeddings,
              continuous_mappings):
    '''
    :param span_embeddings: FloatTensor of size (num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (num. of docs, num. of RoIs, image embed dim),
    :param span_mask: LongTensor of size (num. of spans,),
    :param image_mask: LongTensor of size (num. of RoIs,),
    :return grounded_scores: FloatTensor of size (num. of span pairs,)
    :return grounded_text_scores: FloatTensor of size (num. of span-region pairs,)
    '''
    self.text_scorer.eval()
    with torch.no_grad():
      T = text_embeddings.size(0)
      N = span_embeddings.size(0)
      L = image_embeddings.size(0)
      region_num = image_mask.sum().to(torch.long)
       
      # Extract text-image mention pair indices
      first_idx, second_idx = zip(*list(combinations(range(N), 2)))

      # Compute average span mappings
      # size: (N, M, T)
      continuous_mappings = continuous_mappings.sum(dim=1)
      continuous_mappings = torch.where(continuous_mappings > 0, continuous_mappings, torch.tensor(-9e-9, device=text_embeddings.device))
      # size: (N, T)
      continuous_mappings = F.softmax(continuous_mappings, dim=-1) 

      # Compute grounding scores
      # size: (T, L)
      grounding_scores = torch.matmul(text_embeddings, image_embeddings.t()) 
      grounding_scores = F.softmax(grounding_scores, dim=-1)
      # size: (N, L)
      grounding_scores = torch.mm(continuous_mappings, grounding_scores)

      # Compute text scores      
      text_scores = self.score_text(span_embeddings[first_idx].unsqueeze(0), span_embeddings[second_idx].unsqueeze(0)) 
      text_scores = F.tanh(text_scores)

      # Compute image scores
      # size: (N, D)
      avg_image_embeddings = torch.mm(grounding_scores, image_embeddings)
      image_scores = self.score_image(image_embeddings[first_idx], image_embeddings[second_idx])

      # Compute grounded text scores
      grounded_text_scores = image_scores + text_scores      
    return grounded_text_scores, text_scores, grounding_scores
