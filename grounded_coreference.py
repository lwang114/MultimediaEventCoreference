import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from models import SimplePairWiseClassifier  
    
class GroundedCoreferencer(nn.Module):
  def __init__(self, config):
    super(GroundedCoreferencer, self).__init__()
    self.text_scorer = SimplePairWiseClassifier(config) 
    self.image_scorer = self.score_image
    self.text_only_decode = config.get('text_only_decode', False)

  def score_image(self, first, second, first_mask, second_mask, score_type='both'):
    '''
    :param first: FloatTensor of size (batch size, max num. of first spans, span embed dim)
    :param second: FloatTensor of size (batch size, max num. of second spans, span embed dim)
    :param score_type: str from {'first', 'both'}
    :return scores: FloatTensor of size (batch size, max num. of [score_type] spans, span embed dim)
    '''
    mask = second_mask.unsqueeze(1) * first_mask.unsqueeze(-1)
    att_weights = torch.matmul(first, second.permute(0, 2, 1)) * mask
    att_weights = torch.where(att_weights != 0, att_weights, torch.tensor(-1e10, device=first.device))
    att_weights_first = F.softmax(att_weights, dim=-1) * mask
    att_first = torch.matmul(att_weights_first, second)
    score = F.mse_loss(first, att_first) 
    if score_type == 'both':
      att_weights_second = F.softmax(torch.transpose(att_weights, 1, 2), dim=-1) * torch.transpose(mask, 1, 2)
      att_second = torch.matmul(att_weights_second, first)
      score = score + F.mse_loss(second, att_second)   
    return score, att_first

  def forward(self, span_embeddings, image_embeddings, span_mask, image_mask): # entity_mappings, trigger_mappings,
    '''
    :param span_embeddings: FloatTensor of size (batch size, max num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (batch size, max num. of ROIs, image embed dim), 
    :param entity_mappings: LongTensor of size (batch size, max num. of entities, num. of head entity mentions for coreference),
    :param event_mappings: LongTensor of size (batch size, max num. of head event mentions for coreference),
    :return score: FloatTensor of size (batch size,),
    '''
    self.text_scorer.train()
    loss = self.calculate_loss(span_embeddings, image_embeddings, span_mask, image_mask)
    return loss

  def calculate_loss(self, span_emb, image_emb, span_mask, image_mask):
    B = span_emb.size(0)
    # Compute visual grounding scores   
    loss, _ = self.image_scorer(span_emb, image_emb, span_mask, image_mask)
    return loss
    
  def predict(self, first_span_embeddings, first_image_embeddings, 
              first_span_mask, first_image_mask,
              second_span_embeddings, second_image_embeddings,
              second_span_mask, second_image_mask):
    '''
    :param {first, second}_span_embeddings: FloatTensor of size (num. of spans, span embed dim),
    :param {first, second}_image_embeddings: FloatTensor of size (num. of ROIs, image embed dim),
    :param {first, second}_span_mask: LongTensor of size (max num. of spans,),
    :param {first, second}_image_mask: LongTensor of size (max num. of ROIs,)
    '''
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
      _, first_span_image_emb = self.image_scorer(first_span_embeddings, first_image_embeddings, first_span_mask, first_image_mask)
      _, second_span_image_emb = self.image_scorer(second_span_embeddings, second_image_embeddings, second_span_mask, second_image_mask)
    
    first_span_embeddings = first_span_embeddings.squeeze(0) 
    first_span_image_emb = first_span_image_emb.squeeze(0)
    second_span_embeddings = second_span_embeddings.squeeze(0)
    second_span_image_emb = second_span_image_emb.squeeze(0)
    
    scores = self.text_scorer(first_span_embeddings, second_span_embeddings)
    # print('first span emb: {}'.format(first_span_embeddings[0][1024:2048])) # XXX
    # print('first_span_image_emb: {}'.format(first_span_image_emb[:, 1024:2048])) # XXX
    # first_span_emb = torch.cat([first_span_embeddings[:, :1024], first_span_embeddings[:, :1024], first_span_embeddings[:, :1024], first_span_embeddings[:, -20:]], dim=1) 
    # self_scores = self.text_scorer(10*first_span_emb, 10*first_span_emb)
    # print('self score: {}'.format(self_scores))
    # self_scores2 = self.text_scorer(first_span_image_emb, first_span_image_emb)
    # print('self score img: {}'.format(self_scores2))
    
    if not self.text_only_decode:
      scores = scores + self.text_scorer(first_span_image_emb, second_span_image_emb)
    return scores

  def predict_cluster(self, first_span_embeddings, first_image_embeddings, 
              first_span_mask, first_image_mask,
              second_span_embeddings, second_image_embeddings,
              second_span_mask, second_image_mask):
    '''
    :param {first, second}_span_embeddings: FloatTensor of size (num. of spans, span embed dim),
    :param {first, second}_image_embeddings: FloatTensor of size (num. of ROIs, image embed dim),
    :param {first, second}_span_mask: LongTensor of size (max num. of spans,)
    :param {first, second}_image_mask: LongTensor of size (max num. of ROIs,)
    :return scores: FloatTensor of size (batch size, max num. of mention pairs),
    :return clusters: dict of list of int, mapping from cluster id to mention ids of its members  
    '''
    thres = -0.5 # TODO Make this a config parameter
    scores = self.predict(first_span_embeddings, first_image_embeddings, 
              first_span_mask, first_image_mask,
              second_span_embeddings, second_image_embeddings,
              second_span_mask, second_image_mask)
    span_num = first_span_mask.sum(0).data
    scores = scores.cpu().detach().numpy()
    children = -1 * np.ones(span_num, dtype=np.int64)
    # Antecedent prediction
    for second_idx in range(span_num):
      candidate_scores = []
      for first_idx in range(second_idx):
        score_idx = first_idx * (2 * span_num - first_idx + 1) / 2
        score_idx += second_idx
        score = scores[i, j]
        if children[i] == -1:
          candidate_scores.append(score)
        else:
          candidate_scores.append(-np.inf)
      candidate_scores = np.asarray(candidate_scores)
      max_score = candidate_scores.max()
      if max_score > thres:
        parent = np.argmax(candidate_scores)
        children[parent] = second_idx
    
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
          clusters[cluster_id].append(cur_idx)
          cur_idx = children[cur_idx]
        cluster_id += 1
    print('Number of non-singleton clusters: {}'.format(cluster_id))
    return clusters, scores
