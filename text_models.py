import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import allennlp.nn.util as util 
from itertools import combinations
from transformers import AutoModel
from IPOT.ipot import ipot_WD 

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)

class NoOp(nn.Module):
  def __init__(self):
    super(NoOp, self).__init__()

  def forward(self, x):
    return x

class BiLSTM(nn.Module):
  def __init__(self, input_dim, embedding_dim, num_layers=1, char_dim=0, char_vec=None):
    super(BiLSTM, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_layers = num_layers
    self.batchnorm1 = nn.BatchNorm2d(1) 
    self.cnn = nn.Conv2d(1, input_dim, kernel_size=(input_dim, 3), stride=(1,1), padding=(0,1))
    self.rnn = nn.LSTM(input_size=input_dim+char_dim, hidden_size=embedding_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
    if not char_vec is None:
      self.batchnorm2 = nn.BatchNorm2d(char_vec.shape[1])
      self.char_emb = nn.Embedding(char_vec.shape[0], char_vec.shape[1])
      self.char_emb.weight.data.copy_(torch.from_numpy(char_vec))
      self.char_cnn = nn.Conv2d(char_vec.shape[1], char_dim, kernel_size=(3, 1), stride=(1,1), padding=(0,0))

  def forward(self, x, save_features=False, char_idxs=None):
    if x.dim() < 3:
      x = x.unsqueeze(0)
    B = x.size(0)
    T = x.size(1)
    
    x = x.unsqueeze(1)
    x = torch.transpose(x, -2, -1)
    x = self.batchnorm1(x)
    x = F.relu(self.cnn(x)).squeeze(2).transpose(-2, -1)
    if not char_idxs is None:
      x_char = self.char_emb(char_idxs).permute(0, 3, 2, 1)
      x_char = self.batchnorm2(x_char)
      x_char = F.relu(self.char_cnn(x_char)).max(dim=-2)[0]
      x_char = torch.transpose(x_char, -2, -1)
      x = torch.cat([x, x_char], dim=-1)
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

class CharCNN(nn.Module):
  def __init__(self, input_dim, embedding_dim, char_vec):
    super(CharCNN, self).__init__()
    self.char_dim = char_vec.shape[1]
    self.embedding_dim = embedding_dim

    # Initialize the character embeddings
    self.char_emb = nn.Embedding(char_vec.shape[0], char_vec.shape[1])
    self.char_emb.weight.data.copy_(torch.from_numpy(char_vec))

    self.batchnorm1 = nn.BatchNorm2d(self.char_dim)
    self.conv1 = nn.Conv2d(self.char_dim, 100, kernel_size=(input_dim, 3), stride=(1,1), padding=(0,0))
    self.conv2 = nn.Conv2d(100, 200, kernel_size=(1,3), stride=(1,1), padding=(0,1))
    self.conv3 = nn.Conv2d(200, embedding_dim, kernel_size=(1,3), stride=(1,1), padding=(0,2))
    
  def forward(self, x):
    if x.ndim < 3:
      x = x.unsqueeze(0)
    
    # Shape: batch size x num. of tokens x num. of chars x embedding dim 
    x = self.char_emb(x)

    # Shape: batch size x embedding dim x num. of chars x num. of tokens 
    x = x.permute(0, 3, 2, 1)
    
    x = self.batchnorm1(x)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.squeeze(2)
    x = torch.transpose(x, -2, -1)
    return x 

class BERTSpanEmbedder(nn.Module):
    def __init__(self, config, device):
        super(BERTSpanEmbedder, self).__init__()
        self.bert_model = AutoModel.from_pretrained(config.bert_model)
        self.bert_hidden_size = config.bert_hidden_size 
        self.with_width_embedding = config.with_mention_width
        self.use_head_attention = config.with_head_attention
        self.device = device
        self.dropout = config.dropout
        self.self_attention_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.bert_hidden_size, config.hidden_layer),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config.hidden_layer, 1)
        )
        self.self_attention_layer.apply(init_weights)
        self.width_feature = nn.Embedding(5, config.embedding_dimension)
        self.linguistic_feature_types = config.linguistic_feature_types
    
    def pad_continuous_embeddings(self, continuous_embeddings, width):
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
  
    def forward(self, input_ids,
                start_mappings,
                end_mappings,
                continuous_mappings,
                width,
                linguistic_labels=None,
                attention_mask=None):
        doc_embeddings = self.bert_model(input_ids=input_ids,
                                         attention_mask=attention_mask)
        start_embeddings = torch.matmul(start_mappings, doc_embeddings)
        end_embeddings = torch.matmul(end_mappings, doc_embeddings)
        start_end = torch.cat([start_embeddings, end_embeddings], dim=-1)
        continuous_embeddings = torch.matmul(continuous_mappings, doc_embeddings.unsqueeze(1))
        
        if self.with_start_end_embedding:
          vector = start_end
        else:
          vector = start_embeddings
        
        B, S, M = None, None, None
        if not isinstance(continuous_embeddings, list):
          B = continuous_embeddings.size(0)
          S = continuous_embeddings.size(1)
          M = continuous_embeddings.size(2)
          continuous_embeddings = continuous_embeddings.view(B*S, M, -1)
          width = width.view(B*S)
          vector = vector.view(B*S, -1)
          linguistic_labels = linguistic_labels.view(B*S, -1)

        if self.use_head_attention:
            padded_tokens_embeddings, masks = self.pad_continous_embeddings(continuous_embeddings, width)
            attention_scores = self.self_attention_layer(padded_tokens_embeddings).squeeze(-1)
            attention_scores *= masks
            attention_scores = torch.where(attention_scores != 0, attention_scores,
                                           torch.tensor(-9e9, device=self.device))
            attention_scores = F.softmax(attention_scores, dim=1)
            weighted_sum = (attention_scores.unsqueeze(-1) * padded_tokens_embeddings).sum(dim=1)
            vector = torch.cat((vector, weighted_sum), dim=1)

        for feat_idx, feat_type in enumerate(self.linguistic_feature_types):
            vector = torch.cat((vector, linguistic_labels[:, feat_idx:feat_idx+1]), dim=1)  

        if self.with_width_embedding:
            width = torch.clamp(width, max=4)
            width_embedding = self.width_feature(width)
            vector = torch.cat((vector, width_embedding), dim=1)

        if not isinstance(continuous_embeddings, list):
          vector = vector.view(B, S, -1)
        return vector


class SpanEmbedder(nn.Module):
    def __init__(self, config, device):
        super(SpanEmbedder, self).__init__()
        self.bert_hidden_size = config.bert_hidden_size
        self.with_start_end_embedding = config.with_start_end_embedding
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
        self.linguistic_feature_types = config.linguistic_feature_types

    def pad_continous_embeddings(self, continuous_embeddings, width):
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


    def forward(self, doc_embeddings, 
                start_mappings, 
                end_mappings, 
                continuous_mappings, 
                width,
                linguistic_labels=None,
                attention_mask=None):
        start_embeddings = torch.matmul(start_mappings, doc_embeddings)
        end_embeddings = torch.matmul(end_mappings, doc_embeddings)
        start_end = torch.cat([start_embeddings, end_embeddings], dim=-1)
        continuous_embeddings = torch.matmul(continuous_mappings, doc_embeddings.unsqueeze(1))
       
        if self.with_start_end_embedding: 
          vector = start_end
        else:
          vector = start_embeddings

        B, S, M = None, None, None
        if not isinstance(continuous_embeddings, list):
          B = continuous_embeddings.size(0)
          S = continuous_embeddings.size(1)
          M = continuous_embeddings.size(2)
          continuous_embeddings = continuous_embeddings.view(B*S, M, -1)
          width = width.view(B*S)
          vector = vector.view(B*S, -1)
          linguistic_labels = linguistic_labels.view(B*S, -1)

        if self.use_head_attention:
            padded_tokens_embeddings, masks = self.pad_continous_embeddings(continuous_embeddings, width)
            attention_scores = self.self_attention_layer(padded_tokens_embeddings).squeeze(-1)
            attention_scores *= masks
            attention_scores = torch.where(attention_scores != 0, attention_scores,
                                           torch.tensor(-9e9, device=self.device))
            attention_scores = F.softmax(attention_scores, dim=1)
            weighted_sum = (attention_scores.unsqueeze(-1) * padded_tokens_embeddings).sum(dim=1)
            vector = torch.cat((vector, weighted_sum), dim=1)

        for feat_idx, feat_type in enumerate(self.linguistic_feature_types):
            vector = torch.cat((vector, linguistic_labels[:, feat_idx:feat_idx+1]), dim=1)  

        if self.with_width_embedding:
            width = torch.clamp(width, max=4)
            width_embedding = self.width_feature(width)
            vector = torch.cat((vector, width_embedding), dim=1)

        if not isinstance(continuous_embeddings, list):
          vector = vector.view(B, S, -1)
        return vector

class SpanScorer(nn.Module):
    def __init__(self, config):
        super(SpanScorer, self).__init__()
        self.input_layer = config.bert_hidden_size * 3
        if config.with_mention_width:
            self.input_layer += config.embedding_dimension
        self.mlp = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_layer, config.hidden_layer),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config.hidden_layer, 1)
        )
        self.mlp.apply(init_weights)


    def forward(self, span_embedding):
        return self.mlp(span_embedding)

class SimplePairWiseClassifier(nn.Module):
    def __init__(self, config):
        super(SimplePairWiseClassifier, self).__init__()
        self.input_layer = config.bert_hidden_size
        if config.with_start_end_embedding:
          self.input_layer += config.bert_hidden_size 
        if config.with_head_attention:
          self.input_layer += config.bert_hidden_size 
        
        self.input_layer += len(config.linguistic_feature_types)

        if config.with_mention_width:
          self.input_layer += config.embedding_dimension

        self.input_layer *= 3
        self.hidden_layer = config.hidden_layer
        self.pairwise_mlp = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_layer, self.hidden_layer),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, 1),
        )
        self.pairwise_mlp.apply(init_weights)

    def forward(self, first, second):
        return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1))

    def predict_cluster(self, scores, first_idx, second_idx):
      '''
      :param span_embeddings: FloatTensor of size (num. of spans, span embed dim),
      :param first_idx: LongTensor of size (num. of mention pairs,)
      :param second_idx: LongTensor of size (num. of mention pairs,)
      :param scores: FloatTensor of size (batch size, max num. of mention pairs),
      :return antecedents: int array of size (batch size, num. of mentions), 
      :return clusters: dict of list of int, mapping from cluster id to mention ids of its members
      '''
      device = scores.device
      thres = 0.
      span_num = max(second_idx) + 1
      span_mask = torch.ones(len(first_idx)).to(device)
      antecedents = -1 * np.ones(span_num, dtype=np.int64)

      # Antecedent prediction
      for idx2 in range(span_num):
        candidate_scores = []
        for idx1 in range(idx2):
          score_idx = idx1 * (2 * span_num - idx1 - 1) // 2 - 1
          score_idx += (idx2 - idx1)
          score = scores[score_idx].squeeze().cpu().detach().data.numpy()
          candidate_scores.append(score)

        if len(candidate_scores) > 0:
          candidate_scores = np.asarray(candidate_scores)
          max_score = candidate_scores.max()
          if max_score > thres:
            antecedent = np.argmax(candidate_scores)
            antecedents[idx2] = antecedent      
      return antecedents  

if __name__ == '__main__':
  import argparse
  import pyhocon
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='configs/config_grounded.json')
  args = parser.parse_args()
  
  config = pyhocon.ConfigFactory.parse_file(args.config) 
  # clf = SimplePairWiseClassifier(config)
  clf = SelfAttentionPairWiseClassifier(config)
  # clf.load_state_dict(torch.load(config.pairwise_scorer_path))
  clf.eval()
  with torch.no_grad():
    first = torch.ones((1, clf.input_layer))
    second = torch.ones((1, clf.input_layer))
    print('Pairwise classifier score between all-one vectors: {}'.format(clf(first, second)))
    third = torch.zeros((1, clf.input_layer))
    print('Pairwise classifier score between all-one and all-zero vectors: {}'.format(clf(first, third)))
    first = 0.01*torch.randn((1, clf.input_layer))
    second = 0.01*torch.randn((1, clf.input_layer))
    print('Pairwise classifier score between two random vectors: {}'.format(clf(first, second)))
    print('Pairwise classifier score between random vector and itself: {}'.format(clf(first, first)))
  # A = np.asarray([[1, 0, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
  # print(find_connected_components(A))
