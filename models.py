import torch.nn as nn
import torch
import torch.nn.functional as F



def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)

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



    def forward(self, start_end, continuous_embeddings, width): # XXX
        vector = start_end
        B, S, M = None, None, None
        if not isinstance(continuous_embeddings, list):
          B = continuous_embeddings.size(0)
          S = continuous_embeddings.size(1)
          M = continuous_embeddings.size(2)
          continuous_embeddings = continuous_embeddings.view(B*S, M, -1)
          width = width.view(B*S)
          vector = vector.view(B*S, -1)

        if self.use_head_attention:
            padded_tokens_embeddings, masks = self.pad_continous_embeddings(continuous_embeddings, width)
            attention_scores = self.self_attention_layer(padded_tokens_embeddings).squeeze(-1)
            attention_scores *= masks
            attention_scores = torch.where(attention_scores != 0, attention_scores,
                                           torch.tensor(-9e9, device=self.device))
            attention_scores = F.softmax(attention_scores, dim=1)
            weighted_sum = (attention_scores.unsqueeze(-1) * padded_tokens_embeddings).sum(dim=1)
            vector = torch.cat((vector, weighted_sum), dim=1)

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
        self.input_layer = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2 
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

'''
class BiLSTMSpanEmbedder(nn.Module):
  def __init__(self, config): # TODO
    super(BiLSTMSpanEmbedder, self).__init__()
    
  def forward(doc_embeddings, candidate_start_ends, width, span_mask): # TODO
    lstm_embeddings = self.lstm_embedder(doc_embeddings)
    start_vector = torch.gather(lstm_embeddings, candidate_start_ends[:, :, 0], dim=1)
    end_vector = torch.gather(lstm_embeddings, candidate_start_ends[:, :, 1], dim=1)
    vector = torch.cat([start_vector, end_vector], dim=-1)
    continuous_embeddings = self.get_all_token_embedding(embedding, start, end)
    if self.use_head_attention:
        padded_tokens_embeddings, masks = self.pad_continous_embeddings(continuous_embeddings, width)
        attention_scores = self.self_attention_layer(padded_tokens_embeddings).squeeze(-1)
        attention_scores *= masks
        attention_scores = torch.where(attention_scores != 0, attention_scores,
                                           torch.tensor(-9e9, device=self.device))
        attention_scores = F.softmax(attention_scores, dim=1)
        weighted_sum = (attention_scores.unsqueeze(-1) * padded_tokens_embeddings).sum(dim=1)
        vector = torch.cat((vector, weighted_sum), dim=1)

    if self.with_width_embedding:
        width = torch.clamp(width, max=4)
        width_embedding = self.width_feature(width)
        vector = torch.cat((vector, width_embedding), dim=1)

    vector = vector.view(B, S, -1)
    return vector


  
  def get_all_token_embedding(embedding, start, end): # TODO
    span_embeddings, length = [], []
    for s, e in zip(start, end):
        indices = torch.tensor(range(s, e + 1))
        span_embeddings.append(embedding[indices])
        length.append(len(indices))
    return span_embeddings, length
'''
  

class SymbolicPairWiseClassifier(nn.Module):
  def __init__(self, config):
    super(SymbolicPairWiseClassifier, self).__init__()
    self.symbolic_layer = config.get('type_embedding_dimension', 100)
    self.input_layer = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2 
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
    self.symbolic_pairwise_mlp = nn.Bilinear(self.symbolic_layer, self.symbolic_layer, 1)

    self.pairwise_mlp.apply(init_weights)
    self.symbolic_pairwise_mlp.apply(init_weights)

  def forward(self, first, second, first_symbolic=None, second_symbolic=None):
    if first_symbolic is not None and second_symbolic is not None:
      return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1)) + self.symbolic_pairwise_mlp(first_symbolic, second_symbolic)
    else:
      return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1))
    

if __name__ == '__main__':
  import argparse
  import pyhocon
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='configs/config_grounded_text_only_decode.json')
  args = parser.parse_args()
  
  config = pyhocon.ConfigFactory.parse_file(args.config) 
  clf = SimplePairWiseClassifier(config)
  clf.load_state_dict(torch.load(config.pairwise_scorer_path))
  clf.eval()
  with torch.no_grad():
    first = torch.ones((1, clf.input_layer // 3))
    second = torch.ones((1, clf.input_layer // 3))
    print('Pairwise classifier score between all-one vectors: {}'.format(clf(first, second)))
    third = torch.zeros((1, clf.input_layer // 3))
    print('Pairwise classifier score between all-one and all-zero vectors: {}'.format(clf(first, third)))
    first = 0.01*torch.randn((1, clf.input_layer // 3))
    second = 0.01*torch.randn((1, clf.input_layer // 3))
    print('Pairwise classifier score between two random vectors: {}'.format(clf(first, second)))
    print('Pairwise classifier score between random vector and itself: {}'.format(clf(first, first)))
