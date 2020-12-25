import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
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
    
class SupervisedGroundedCoreferencer(nn.Module):
  def __init__(self, config):
    super(SupervisedGroundedCoreferencer, self).__init__()
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
    # mapping = torch.where(mask > 0, mask, torch.tensor(-9e9, device=first.device))
    # mapping = F.softmax(mapping.view(B, B, N, L), dim=-1)
    scores = (scores * mask).view(B, B, N, L)
    return scores

  def calculate_loss(self, span_emb, image_emb, span_mask, image_mask):
    B = span_emb.size(0)
    # Compute visual grounding scores
    scores = self.score_grounding(span_emb, image_emb, span_mask, image_mask)
    return scores
  
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
    scores = self.calculate_loss(span_embeddings,
                                 image_embeddings,
                                 span_mask,
                                 image_mask)
  
    S = scores.sum(-1).max(-1)[0]
    loss = -torch.sum(m(S))-torch.sum(m(S.transpose(0, 1)))
    loss = loss / n
    return loss, scores
    
  def retrieve(self, span_embeddings, image_embeddings, span_mask, image_mask, k=10):
    '''
    :param span_embeddings: FloatTensor of size (num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (num. of ROIs, image embed dim),
    '''
    self.text_scorer.eval()
    n = span_embeddings.size(0)
    span_embeddings = span_embeddings.cpu()
    image_embeddings = image_embeddings.cpu()
    span_mask = span_mask.cpu()
    image_mask = image_mask.cpu()
    scores = self.predict(span_embeddings, image_embeddings, span_mask, image_mask)
    S = scores.sum(-1).max(-1)[0]
    _, I2S_idxs = S.topk(k, 0)
    _, S2I_idxs = S.topk(k, 1) 
    return I2S_idxs.t(), S2I_idxs
    
  def predict(self, span_embeddings, image_embeddings, 
              span_mask, image_mask):
    if span_embeddings.ndim == 2:
      span_embeddings = span_embeddings.unsqueeze(0)
      image_embeddings = image_embeddings.unsqueeze(0)  
      span_mask = span_mask.unsqueeze(0)
      image_mask = image_mask.unsqueeze(0)

    self.text_scorer.eval()
    self.grounding_scorer.eval()
    with torch.no_grad():
      scores = self.score_grounding(span_embeddings, image_embeddings, span_mask, image_mask)
    
    return scores
