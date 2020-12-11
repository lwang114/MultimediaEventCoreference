import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from models import SimplePairWiseClassifier, SymbolicPairWiseClassifier 

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

  def forward(self, x, mask=None, return_feat=False)
    if return_feat:
      return x, x, mask
    else:
      return x, mask

class GaussianSoftmax(nn.Module):
  def __init__(self, in_features, out_features):
    super(GaussianSoftmax, self).__init__()
    self.K = out_features
    self.d = in_features
    self.codebook = nn.Parameter(torch.empty(in_features, out_features, requires_grad=True))
    nn.init.xavier_uniform_(self.codebook)

  def forward(self, x):
    if x.ndim() <= 2:
      x = x.unsqueeze(0) 
    score = - torch.mean((x.unsqueeze(-2) - self.codebook) ** 2, dim=-1) 
    out = F.softmax(score)
    return out


class SMTGroundedCoreferencer(nn.Module):
  def __init__(self, config):
    super(GroundedCoreferencer, self).__init__()
    self.coref_scorer = SymbolicPairWiseClassifier(config) 
    self.input_layer = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2 
    if config.with_mention_width:
      self.input_layer += config.embedding_dimension
    
    self.text_scorer = self.GaussianSoftmax(self.input_layer, config.n_text_clusters)
    self.image_scorer = self.GaussianSoftmax(self.input_layer, config.n_image_clusters)
    self.grounding_scorer = self.score_grounding
    self.translate_logits = nn.Parameter(torch.empty(config.n_text_clusters, config.n_image_clusters, ), requires_grad=True)
    nn.init.xavier_uniform_(self.translate_logits)
    
    self.text_only_decode = config.get('text_only_decode', False)

  def score_image(self, first, second, first_mask, second_mask, score_type='both'):
    '''
    :param first: FloatTensor of size (batch size, max num. of first spans, span embed dim)
    :param second: FloatTensor of size (batch size, max num. of second spans, span embed dim)
    :param score_type: str from {'first', 'both'}
    :return scores: FloatTensor of size (batch size, max num. of [score_type] spans, span embed dim)
    '''
    translate_prob = F.softmax(self.translate_logits)
    concept_probs = torch.matmul(first, translate_prob)
    aligns_probs = torch.matmul(concept_probs, second.permute(0, 2, 1)))
    first_map = F.softmax(1e14*first_mask)
    align_probs = torch.matmul(first_map.unsqueeze(-2), aligns_probs, dim=1)
    nll = -torch.matmul(torch.log(align_probs), second_mask.unsqueeze(-1)).squeeze(-1)
    return nll, concept_probs

  def forward(self, span_embeddings, image_embeddings, span_mask, image_mask): # entity_mappings, trigger_mappings,
    '''
    :param span_embeddings: FloatTensor of size (batch size, max num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (batch size, max num. of ROIs, image embed dim), 
    :param entity_mappings: LongTensor of size (batch size, max num. of entities, num. of head entity mentions for coreference),
    :param event_mappings: LongTensor of size (batch size, max num. of head event mentions for coreference),
    :return score: FloatTensor of size (batch size,),
    '''
    self.image_scorer.train()
    self.text_scorer.train()
    self.coref_scorer.train()
    n = span_embeddings.size(0)
    S = torch.zeros((n, n), dtype=torch.float, device=span_embeddings.device)
    for s_idx in range(n):
      for v_idx in range(n):
        S[s_idx, v_idx] = self.calculate_loss(span_embeddings[s_idx].unsqueeze(0),
                                              image_embeddings[v_idx].unsqueeze(0),
                                              span_mask[s_idx].unsqueeze(0),
                                              image_mask[v_idx].unsqueeze(0))
    loss = torch.mean(S.diag().unsqueeze(-1) - S) + torch.mean(S.diag().unsqueeze(-1) - S.t())
    return loss

  def calculate_loss(self, span_emb, image_emb, span_mask, image_mask):
    B = span_emb.size(0)
    # Compute visual grounding scores   
    losses , _ = self.grounding_scorer(span_emb, image_emb, span_mask, image_mask) 
    return torch.mean(losses)
    
  def predict(self, first_span_embeddings, first_image_embeddings, 
              first_span_mask, first_image_mask,
              second_span_embeddings, second_image_embeddings,
              second_span_mask, second_image_mask):
    '''
    :param span_embeddings: FloatTensor of size (num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (num. of ROIs, image embed dim),
    '''
    first_span_embeddings = first_span_embeddings.unsqueeze(0)
    first_image_embeddings = first_image_embeddings.unsqueeze(0)  
    first_span_mask = first_span_mask.unsqueeze(0)
    first_image_mask = first_image_mask.unsqueeze(0)
    second_span_embeddings = second_span_embeddings.unsqueeze(0)
    second_image_embeddings = second_image_embeddings.unsqueeze(0)
    second_span_mask = second_span_mask.unsqueeze(0)
    second_image_mask = second_image_mask.unsqueeze(0)
    
    self.coref_scorer.eval()
    self.image_scorer.eval()
    self.text_scorer.eval()
    with torch.no_grad(): # TODO
      first_concept_probs = self.text_scorer(first_span_embeddings)
      second_concept_probs = self.text_scorer(second_span_embeddings)

      first_span_embeddings = first_span_embeddings.squeeze(0) 
      first_concept_probs = first_concept_probs.squeeze(0)
      second_span_embeddings = second_span_embeddings.squeeze(0)
      second_concept_probs = second_concept_probs.squeeze(0)
    
      scores = self.coref_scorer(first_span_embeddings, 
                                 second_span_embeddings, 
                                 first_concept_probs, 
                                 second_concept_probs)
    return scores
