import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from models import SimplePairWiseClassifier  

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
    

class MMLGroundedCoreferencer(nn.Module):
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
    att_weights = torch.matmul(first, second.permute(0, 2, 1))
    att_weights_first = F.softmax(att_weights * second_mask.unsqueeze(1), dim=-1)
    att_first = torch.matmul(att_weights_first, second)
    score = F.mse_loss(first, att_first) 

    if score_type == 'both':
      att_weights_second = F.softmax(torch.transpose(att_weights, 1, 2) * first_mask.unsqueeze(1), dim=-1)    
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
    n = span_embeddings.size(0)
    S = torch.zeros((n, n), dtype=torch.float, device=span_embeddings.device, requires_grad=True)
    m = nn.LogSoftmax(dim=1) 
    for s_idx in range(n):
      for v_idx in range(n):
        S[s_idx, v_idx] = self.calculate_loss(span_embeddings[s_idx], image_embeddings[v_idx], span_mask, image_mask)
    
    loss = -torch.sum(m(S).diag())-torch.sum(m(S.transpose(0, 1)).diag())
    loss = loss / n
    return loss

  def calculate_loss(self, span_emb, image_emb, span_mask, image_mask):
    B = span_emb.size(0)
    # Compute visual grounding scores   
    loss, _ = self.image_scorer(span_emb, image_emb, span_mask, image_mask)
    return loss
    
  def retrieve(self, span_embeddings, image_embeddings, span_mask, image_mask, k=10):
    '''
    :param span_embeddings: FloatTensor of size (num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (num. of ROIs, image embed dim),
    '''
    self.text_scorer.eval()
    n = span_embeddings.size(0)
    span_embeddings = span_embeddings.cpu()
    image_embeddings = image_embeddings.cpu()
    S = torch.zeros((n, n), dtype=torch.float, device=torch.device('cpu'), requires_grad=False)
    m = nn.LogSoftmax(dim=1)
    for s_idx in range(n):
      for v_idx in range(n):
        S[s_idx, v_idx] = -self.calculate_loss(span_embeddings[s_idx], image_embeddings[v_idx], span_mask, image_mask)
    return S.topk(k, 0).t(), S.topk(k, 1) 

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
      _, first_span_image_emb = self.image_scorer(first_span_embeddings, first_image_embeddings, first_span_mask, first_image_mask)
      _, second_span_image_emb = self.image_scorer(second_span_embeddings, second_image_embeddings, second_span_mask, second_image_mask)
    
    first_span_embeddings = first_span_embeddings.squeeze(0) 
    first_span_image_emb = first_span_image_emb.squeeze(0)
    second_span_embeddings = second_span_embeddings.squeeze(0)
    second_span_image_emb = second_span_image_emb.squeeze(0)
    
    scores = self.text_scorer(first_span_embeddings, second_span_embeddings)
    if not self.text_only_decode:
      scores = scores + self.text_scorer(first_span_image_emb, second_span_image_emb)
    return scores
