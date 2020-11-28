import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
    

class ResNet152:
  def __init__(self, embedding_dim=1024, device=torch.device('cpu')):
    net = getattr(models, 'resnet152')(pretrained=True)
    b = list(net.children())
    self.backbone = nn.Sequential(*b[:-2])
    self.pooler = nn.Sequential(*b[-2])
    self.embedder = nn.Linear(2048, embedding_dim)
    
    for p in self.backbone.parameters():
      p.requires_grad = False
    for p in self.pooler.parameters():
      p.requires_grad = False
     
    self.device = device
    self.to(device)
     
  def forward(self, images):    
    fmap = self.backbone(images)
    emb = self.pooler(fmap)
    B, D, H, W = emb.size()
    self.embedder(emb.permute(0, 2, 3, 1).view(B*H*W, D)).view(B, H, W, D)
    return emb
    

class GroundedCoreferencer(nn.Module):
  def __init__(self, config):
    self.text_scorer = SimplePairWiseClassifier(config) 
    self.image_scorer = self.score_image

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
    scores = -torch.dist(first, att_first, 2) 
    if score_type == 'both':
      att_weights_second = F.softmax(torch.transpose(att_weights, 1, 2) * first_mask.unsqueeze(1), dim=-1)    
      att_second = torch.matmul(att_weights_second, first)
      scores = scores - torch.dist(second, att_second, 2)
    return scores, att_first

  def forward(self, span_embeddings, image_embeddings, span_mask, image_mask): # entity_mappings, trigger_mappings,
    '''
    :param span_embeddings: FloatTensor of size (batch size, max num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (batch size, max num. of ROIs, image embed dim), 
    :param entity_mappings: LongTensor of size (batch size, max num. of entities, num. of head entity mentions for coreference),
    :param event_mappings: LongTensor of size (batch size, max num. of head event mentions for coreference),
    :return score: FloatTensor of size (batch size,),
    '''
    self.text_scorer.train()
    self.image_scorer.train()
    loss = self.calculate_loss(span_embeddings, image_embeddings, span_mask, image_mask)
    return loss

  def calculate_loss(self, span_emb, image_emb, span_mask, image_mask):
    B = span_emb.size(0)
    # Compute visual grounding scores   
    image_scores, _ = self.image_scorer(span_emb, image_emb, span_mask, image_mask)
    return torch.mean(image_scores)
    
  def predict(self, first_span_embeddings, first_image_embeddings, 
              first_span_mask, first_image_mask,
              second_span_embeddings, second_image_embeddings,
              second_span_mask, second_image_mask, batched=True):
    '''
    :param span_embeddings: FloatTensor of size (num. of spans, span embed dim),
    :param image_embeddings: FloatTensor of size (num. of ROIs, image embed dim),
    :param delta: FloatTensor of size (num. of spans,)
    :param back_ptrs: LongTensor of size (num. of spans,) 
    '''
    if not batched:
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
    
    scores = self.text_scorer(first_span_embeddings, second_span_embeddings) +\
             self.text_scorer(first_span_image_emb, second_span_image_emb)
    return scores
