import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from itertools import combinations
from models import SimplePairWiseClassifier  

class BiLSTM(nn.Module):
  def __init__(self, input_dim, embedding_dim, num_layers=1):
    super(BiLSTM, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_layers = num_layers
    self.rnn = nn.LSTM(input_size=input_dim, hidden_size=embedding_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

  def forward(self, x, save_features=False):
    if x.dim() < 3:
      x = x.unsqueeze(0)
    B = x.size(0)
    T = x.size(1)
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
    
class JointSupervisedGroundedCoreferencer(nn.Module):
  def __init__(self, config):
    super(JointSupervisedGroundedCoreferencer, self).__init__()
    self.text_scorer = SimplePairWiseClassifier(config) 
    self.grounding_scorer = SimplePairWiseClassifier(config)
    self.text_only_decode = config.get('text_only_decode', False)

  def score_grounding(self, first, second, first_mask, second_mask, score_type='both'):
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
    first = first.view(B*N, -1)
    second = second.view(B*L, -1)
    first_mask = first_mask.view(B*N, 1)
    second_mask = second_mask.view(B*L, 1)
    
    # Compute index pairs for all combinations between first and second
    first_idxs = [i for i in range(B*N) for j in range(B*L)] 
    second_idxs = [j for i in range(B*N) for j in range(B*L)] 
    first_idxs = torch.LongTensor(first_idxs)
    second_idxs = torch.LongTensor(second_idxs)
    mask = first_mask[first_idxs] * second_mask[second_idxs]
    
    # Compute grounding scores
    scores = self.grounding_scorer(first[first_idxs], second[second_idxs])
    scores = (scores * mask).view(B, B, N, L)
    
    return scores
  
  def score_text(self, span_embeddings, span_mask):
    B = span_embeddings.size(0)
    N = span_embeddings.size(1)
    first_idx, second_idx = zip(*list(combinations(range(N), 2)))
    first = span_embeddings[:, first_idx].view(B*len(first_idx), -1)
    second = span_embeddings[:, second_idx].view(B*len(second_idx), -1)
    scores = self.text_scorer(first, second)
    scores = scores.view(B, len(first_idx))
    return scores
  
  def score_image(self, image_embeddings, image_mask):
    scores = torch.matmul(image_embeddings, image_embeddings.t())
    norm_factor = torch.norm(image_embeddings, dim=-1).unsqueeze(-1) * torch.norm(image_embeddings, dim=-1).unsqueeze(-2)  
    scores = scores / norm_factor
    return scores

  def calculate_loss(self, span_emb, image_emb, span_mask, image_mask):
    B = span_emb.size(0)
    # Compute visual grounding scores
    grounding_scores = self.score_grounding(span_emb, image_emb, span_mask, image_mask)
    text_scores = self.score_text(span_emb, span_mask)
    return grounding_scores, text_scores 
  
  def forward(self, span_embeddings, image_embeddings, span_mask, image_mask):
    '''
    :param span_embeddings: FloatTensor of size (batch size, max num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (batch size, max num. of ROIs, image embed dim), 
    :param entity_mappings: LongTensor of size (batch size, max num. of entities, num. of head entity mentions for coreference),
    :param event_mappings: LongTensor of size (batch size, max num. of head event mentions for coreference),
    :return score: FloatTensor of size (batch size,),
    '''
    self.text_scorer.train()
    self.grounding_scorer.train()
    n = span_embeddings.size(0)
    m = nn.LogSoftmax(dim=1)
    grounding_scores, text_scores = self.calculate_loss(span_embeddings,
                                 image_embeddings,
                                 span_mask,
                                 image_mask)
  
    S = grounding_scores.sum(-1).sum(-1)
    loss = -torch.sum(m(S))-torch.sum(m(S.transpose(0, 1)))
    loss = loss / n
    return loss, grounding_scores, text_scores
    
  def retrieve(self, span_embeddings, image_embeddings, span_mask, image_mask, k=10):
    '''
    :param span_embeddings: FloatTensor of size (num. of docs, num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (num. of docs, num. of ROIs, image embed dim),
    '''
    self.text_scorer.eval()
    n = span_embeddings.size(0)
    span_embeddings = span_embeddings.cpu()
    image_embeddings = image_embeddings.cpu()
    span_mask = span_mask.cpu()
    image_mask = image_mask.cpu()
    S = torch.zeros((n, n), dtype=torch.float, device=torch.device('cpu'), requires_grad=False)
    m = nn.LogSoftmax(dim=1)
    for s_idx in range(n):
      for v_idx in range(n):
        score = -self.calculate_loss(span_embeddings[s_idx].unsqueeze(0),
                                     image_embeddings[v_idx].unsqueeze(0),
                                     span_mask[s_idx].unsqueeze(0),
                                     image_mask[v_idx].unsqueeze(0))
        S[s_idx, v_idx] = score.sum()
    _, I2S_idxs = S.topk(k, 0)
    _, S2I_idxs = S.topk(k, 1) 
    return I2S_idxs.t(), S2I_idxs
    
  def predict(self, span_embeddings, image_embeddings, 
              span_mask, image_mask): # TODO
    '''
    :param span_embeddings: FloatTensor of size (num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (num. of docs, num. of RoIs, image embed dim),
    :param span_mask: LongTensor of size (num. of spans,),
    :param image_mask: LongTensor of size (num. of RoIs,),
    :return grounded_scores: FloatTensor of size (num. of span pairs,)
    :return grounded_text_scores: FloatTensor of size (num. of span-region pairs,)
    '''
    N = span_embeddings.size(0)
    L = image_embeddings.size(0)
    region_num = image_mask.sum().to(torch.long)
     
    # Extract text-image mention pair indices
    first = [first_idx for first_idx in range(N) for second_idx in range(L)] 
    second = [second_idx for first_idx in range(N) for second_idx in range(L)] 

    self.text_scorer.eval()
    self.grounding_scorer.eval()
    with torch.no_grad():
      # Compute grounding scores
      grounding_scores = self.grounding_scorer(span_embeddings[first], image_embeddings[second]) 
      grounding_scores = grounding_scores.view(N, L)

      # Compute image scores
      image_scores = self.score_image(image_embeddings, image_mask)

      # Compute text scores
      text_scores = self.score_text(span_embeddings.unsqueeze(0), span_mask.unsqueeze(0)).squeeze(0)
    
      # Compute grounded text scores
      first_grounding_scores = F.sigmoid(grounding_scores[first_ground_idx])
      second_grounding_scores = F.sigmoid(grounding_scores[second_ground_idx])
      grounded_pos_scores = 1 + first_grounding_scores.unsqueeze(-1) * second_grounding_scores.unsqueeze(-2) * (image_scores.exp() - 1)
      grounded_pos_scores = (torch.log(grounded_pos_scores[:, :region_num, :region_num])).sum(-1).sum(-1)
      grounded_scores = text_scores + grounded_pos_scores
    return grounded_scores, grounded_text_scores, first_text_idx, second_text_idx, first_ground_idx, second_ground_idx
