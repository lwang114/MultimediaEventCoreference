import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from models import SimplePairWiseClassifier  
import json

class NoOp(nn.Module):
  def __init__(self):
    super(NoOp, self).__init__()

  def forward(self, x):
    return x

class BiLSTM(nn.Module):
  def __init__(self, input_dim, embedding_dim, num_layers=1):
    super(BiLSTM, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_layers = num_layers
    self.batchnorm1 = nn.BatchNorm2d(1) 
    self.cnn = nn.Conv2d(1, input_dim, kernel_size=(input_dim, 3), stride=(1,1), padding=(0,1))
    self.rnn = nn.LSTM(input_size=input_dim, hidden_size=embedding_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

  def forward(self, x, save_features=False):
    if x.dim() < 3:
      x = x.unsqueeze(0)
    B = x.size(0)
    T = x.size(1)
    
    x = x.unsqueeze(1)
    x = torch.transpose(x, -2, -1)
    x = self.batchnorm1(x)
    x = F.relu(self.cnn(x)).squeeze(2).transpose(-2, -1)
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

 
class Davenet(nn.Module):
  def __init__(self, input_dim, embedding_dim=512):
    super(Davenet, self).__init__()
    self.embedding_dim = embedding_dim
    self.batchnorm1 = nn.BatchNorm2d(1)
    self.conv1 = nn.Conv2d(1, 64, kernel_size=(input_dim, 3), stride=(1,1), padding=(0,0))
    self.conv2 = nn.Conv2d(64, 256, kernel_size=(1,3), stride=(1,1), padding=(0,1))
    self.conv3 = nn.Conv2d(256, embedding_dim, kernel_size=(1,3), stride=(1,1), padding=(0,2))
    self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
    
  def forward(self, x):
    if x.ndim < 3:
      x = x.unsqueeze(0)
    if x.dim() == 3:
      x = x.unsqueeze(1)
    x = torch.transpose(x, -2, -1)
    x = self.batchnorm1(x)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.squeeze(2)
    x = torch.transpose(x, -2, -1)
    return x
 
    
class MMLDotProductGroundedCoreferencer(nn.Module):
  def __init__(self, config):
    super(MMLDotProductGroundedCoreferencer, self).__init__()
    self.text_scorer = SimplePairWiseClassifier(config) 
    self.grounding_scorer = self.score_grounding
    self.text_only_decode = config.get('text_only_decode', False)

  def score_grounding(self, first, second, first_mask, second_mask, score_type='both'): 
    '''
    :param first: FloatTensor of size (batch size, max num. of first spans, span embed dim)
    :param second: FloatTensor of size (batch size, max num. of second spans, span embed dim)
    :param score_type: str from {'first', 'both'}
    :return scores: FloatTensor of size (batch size, max num. of [score_type] spans, span embed dim)
    '''
    mask = first_mask.unsqueeze(-1) * second_mask.unsqueeze(1)
    att_weights = torch.matmul(first, second.permute(0, 2, 1)) * mask
    att_weights = torch.where(att_weights != 0, att_weights, torch.tensor(-1e10, device=first.device))

    att_weights_first = F.softmax(att_weights * second_mask.unsqueeze(1), dim=-1) * mask
    att_first = torch.matmul(att_weights_first, second)
    score = - (att_weights_first * att_weights).sum(dim=2).mean() # F.mse_loss(first, att_first)

    if score_type == 'both':
      att_weights_second = F.softmax(torch.transpose(att_weights, 1, 2), dim=-1) * torch.transpose(mask, 1, 2)    
      att_second = torch.matmul(att_weights_second, first)
      score = score - (att_weights_second.permute(0, 2, 1) * att_weights).sum(dim=2).mean() # F.mse_loss(second, att_second)
    
    return score, att_first

  def forward(self, span_embeddings, image_embeddings, span_mask, image_mask, topic_labels=None): # entity_mappings, trigger_mappings,
    '''
    :param span_embeddings: FloatTensor of size (batch size, max num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (batch size, max num. of ROIs, image embed dim), 
    :param entity_mappings: LongTensor of size (batch size, max num. of entities, num. of head entity mentions for coreference),
    :param event_mappings: LongTensor of size (batch size, max num. of head event mentions for coreference),
    :return score: FloatTensor of size (batch size,),
    '''
    self.text_scorer.train()
    n = span_embeddings.size(0)
    device = span_embeddings.device
    S = torch.zeros((n, n), dtype=torch.float, device=device)
    pos_mask = torch.eye(n, dtype=torch.float, device=device)
    m = nn.LogSoftmax(dim=1) 
    for s_idx in range(n):
      for v_idx in range(n):
        S[s_idx, v_idx] = -self.calculate_loss(span_embeddings[s_idx].unsqueeze(0),
                                                 image_embeddings[v_idx].unsqueeze(0),
                                                 span_mask[s_idx].unsqueeze(0),
                                                 image_mask[v_idx].unsqueeze(0))
        if topic_labels is not None:
          pos_mask[s_idx, v_idx] = (topic_labels[s_idx] == topic_labels[v_idx]).to(torch.float).to(device)
    loss = - (m(S) * pos_mask + m(S.t()) * pos_mask).sum()
    loss = loss / n
    return loss

  def calculate_loss(self, span_emb, image_emb, span_mask, image_mask):
    B = span_emb.size(0)
    # Compute visual grounding scores   
    loss, _ = self.grounding_scorer(span_emb, image_emb, span_mask, image_mask)
    return loss
    
  def retrieve(self, span_embeddings, image_embeddings, span_mask, image_mask, k=10):
    '''
    :param span_embeddings: FloatTensor of size (num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (num. of ROIs, image embed dim),
    '''
    self.text_scorer.eval()
    device = torch.device('cpu')
    n = span_embeddings.size(0)
    S = torch.zeros((n, n), dtype=torch.float, device=device, requires_grad=False)
    for s_idx in range(n):
      for v_idx in range(n):
        S[s_idx, v_idx] = -self.calculate_loss(span_embeddings[s_idx].unsqueeze(0),
                                               image_embeddings[v_idx].unsqueeze(0),
                                               span_mask[s_idx].unsqueeze(0),
                                               image_mask[v_idx].unsqueeze(0)).to(device)
    _, I2S_idxs = S.topk(k, 0)
    _, S2I_idxs = S.topk(k, 1) 
    return I2S_idxs.t(), S2I_idxs
    
  def predict(self, first_span_embeddings, first_image_embeddings, 
              first_span_mask, first_image_mask,
              second_span_embeddings, second_image_embeddings,
              second_span_mask, second_image_mask):
    first_span_embeddings = first_span_embeddings.unsqueeze(0)
    first_image_embeddings = first_image_embeddings.unsqueeze(0)  
    first_span_mask = first_span_mask.unsqueeze(0)
    first_image_mask = first_image_mask.unsqueeze(0)
    second_span_embeddings = second_span_embeddings.unsqueeze(0)
    second_image_embeddings = second_image_embeddings.unsqueeze(0)
    second_span_mask = second_span_mask.unsqueeze(0)
    second_image_mask = second_image_mask.unsqueeze(0)
    
    self.text_scorer.eval()
    with torch.no_grad():
      _, first_span_image_emb = self.grounding_scorer(first_span_embeddings, first_image_embeddings, first_span_mask, first_image_mask)
      _, second_span_image_emb = self.grounding_scorer(second_span_embeddings, second_image_embeddings, second_span_mask, second_image_mask)
    
    first_span_embeddings = first_span_embeddings.squeeze(0) 
    first_span_image_emb = first_span_image_emb.squeeze(0)
    second_span_embeddings = second_span_embeddings.squeeze(0)
    second_span_image_emb = second_span_image_emb.squeeze(0)
    
    scores = self.text_scorer(first_span_embeddings, second_span_embeddings)
    if not self.text_only_decode:
      scores = scores + self.text_scorer(first_span_image_emb, second_span_image_emb)
    return scores

  def predict_cluster(self, span_embeddings, image_embeddings, 
                      first_idx, second_idx):
    '''
    :param span_embeddings: FloatTensor of size (num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (num. of ROIs, image embed dim),
    :param first_idx: LongTensor of size (num. of mention pairs,)
    :param second_idx: LongTensor of size (num. of mention pairs,)
    :return scores: FloatTensor of size (batch size, max num. of mention pairs),
    :return clusters: dict of list of int, mapping from cluster id to mention ids of its members 
    '''
    device = span_embeddings.device
    thres = -0.76 # TODO Make this a config parameter
    span_num = max(second_idx) + 1
    span_mask = torch.ones(len(first_idx)).to(device)
    image_mask = torch.ones(image_embeddings.size(0)).to(device) 
    first_span_embeddings = span_embeddings[first_idx]
    second_span_embeddings = span_embeddings[second_idx]

    scores = self.predict(first_span_embeddings, image_embeddings, 
              span_mask, image_mask,
              second_span_embeddings, image_embeddings,
              span_mask, image_mask)
    children = -1 * np.ones(span_num, dtype=np.int64)

    # Antecedent prediction
    for idx2 in range(span_num):
      candidate_scores = []
      for idx1 in range(idx2):
        score_idx = idx1 * (2 * span_num - idx1 - 1) // 2 - 1
        score_idx += (idx2 - idx1)
        score = scores[score_idx].squeeze().cpu().detach().data.numpy()
        if children[idx1] == -1:
          candidate_scores.append(score)
        else:
          candidate_scores.append(-np.inf)

      if len(candidate_scores) > 0:
        candidate_scores = np.asarray(candidate_scores)
        max_score = candidate_scores.max()
        if max_score > thres:
          parent = np.argmax(candidate_scores)
          children[parent] = idx2
    
    # Extract clusters from coreference chains
    cluster_id = 0
    clusters = {}
    covered_spans = []
    for idx in range(span_num):
      if not idx in covered_spans and children[idx] != -1:
        covered_spans.append(idx)
        clusters[cluster_id] = [idx]
        cur_idx = idx
        while children[cur_idx] != -1:
          cur_idx = children[cur_idx]
          clusters[cluster_id].append(cur_idx)
          covered_spans.append(cur_idx)
        cluster_id += 1
    # print('Number of non-singleton clusters: {}'.format(cluster_id))
    return clusters, scores
