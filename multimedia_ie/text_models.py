# TODO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)

class BiLSTM(nn.Module):
  def __init__(self, input_dim, embedding_dim, num_layers=1):
    super(BiLSTM, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_layers = num_layers
    self.rnn = nn.LSTM(input_size=input_dim+char_dim, hidden_size=embedding_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

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
    
class SpanEmbedder(nn.Module):
    def __init__(self, config, device):
        super(SpanEmbedder, self).__init__()
        self.bert_hidden_size = config.bert_hidden_size
        self.with_width_embedding = config.with_mention_width
        self.use_head_attention = config.with_head_attention
        self.device = device
        self.dropout = config.dropout
        self.padded_vector = torch.zeros(self.bert_hidden_size, device=device)
        self.self_attention_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.bert_hidden_size, config.hidden_layer),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config.hidden_layer, 1)
        )
        self.self_attention_layer.apply(init_weights)
        self.width_feature = nn.Embedding(5, config.embedding_dimension)

    def pad_continous_embeddings(self, continuous_embeddings, width): # XXX
        if isinstance(continuous_embeddings, list):
          max_length = max(len(v) for v in continuous_embeddings)
          padded_tokens_embeddings = torch.stack(
            [torch.cat((emb, self.padded_vector.repeat(max_length - len(emb), 1)))
             for emb in continuous_embeddings]
          )
          masks = torch.stack(
            [torch.cat(
                (torch.ones(len(emb), device=self.device),
                 torch.zeros(max_length - len(emb), device=self.device)))
             for emb in continuous_embeddings]
          )
        else:
          span_num = width.size(0)
          max_length = continuous_embeddings.size(1)
          if (max_length - width.max()) < 0:
            print('max width {} is shorter than max width {} in the batch!'.format(max_length, width.max()))
          padded_tokens_embeddings = continuous_embeddings
          masks = torch.stack(
            [torch.cat(
                (torch.ones(max(width[idx].item(), 1), device=self.device),
                 torch.zeros(max_length - max(width[idx].item(), 1), device=self.device)))
             for idx in range(span_num)]
          )

        return padded_tokens_embeddings, masks

class JointClassifier(nn.Module):
  def __init__(self, 
               n_event_types,
               n_role_types,
               n_entity_types,
               config):
    super(JointClassifier, self).__init__()
    self.input_layer = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2
    if config.with_mention_width:
      self.input_layer += config.embedding_dimension

    self.event_classifier = nn.Linear(self.input_layer, n_event_types)
    self.arg_classifier = nn.Linear(self.input_layer*2, n_role_types)  
    self.entity_classifer = nn.Linear(self.input_layer, n_entity_types)

  def forward(self, event_emb, arg_embs):
    n_args = arg_embs.size(1)
    event_scores = self.event_classifier(event_emb)
    entity_scores = self.entity_classifier(arg_embs)
    event_arg_embs = torch.cat([event_emb.repeat(1, n_args, 1), arg_embs], dim=-1)
    arg_scores = self.arg_classifier(event_arg_embs)
    return event_scores, arg_scores, entity_scores
     
