import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import json

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

class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_edge_types,
                 dropout=0.5, bias=True, use_bn=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_edge_types = num_edge_types
        self.dropout = dropout
        self.Gates = nn.ModuleList()
        self.GraphConv = nn.ModuleList()
        self.use_bn = use_bn
        for _ in range(num_edge_types):
            self.Gates.append(nn.Linear(input_dim, 1, bias=bias))
            self.GraphConv.append(nn.Linear(input_dim, embedding_dim, bias=bias))
            nn.init.orthogonal_(self.Gates[-1].weight)
            nn.init.orthogonal_(self.GraphConv[-1].weight)

    def forward(self, x, adj):
        '''
        :param x: FloatTensor, input feature tensor, (batch size, seq len, input dim)
        :param adj: FloatTensor, (sparse.FloatTensor.to_dense()), adjacent matrix, (batch size, edge type, seq len, seq len)
        :return out: FloatTensor, (batch size, seq len, input dim)
        '''
        adj_ = adj.transpose(0, 1)
        B = adj.size(0)
        ts = []
        for i in range(self.num_edge_types):
            gate_status = F.sigmoid(self.Gates[i](x))
            adj_hat_i = adj_[i] * gate_status
            t = torch.bmm(adj_hat_i, self.GraphConv[i](x))
            ts.append(t)
        ts = torch.stack(ts).sum(dim=0)
        
        if self.use_bn:
            ts = ts.transpose(1, 2).contiguous()
            ts = self.bn(ts)
            ts = ts.transpose(1, 2).contiguous()
        out = F.dropout(F.relu(ts), p=self.dropout)
        return out
        
        
class SpanEmbedder(nn.Module):
    def __init__(self, config, device):
        super(SpanEmbedder, self).__init__()
        self.bert_hidden_size = config.bert_hidden_size
        self.with_width_embedding = config.with_mention_width
        self.use_head_attention = config.with_head_attention
        self.with_type_embedding = config.get('with_type_embedding', False)
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
        if self.with_type_embedding:
            self.type_feature = nn.Embedding(200, config.type_embedding_dimension)


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


    def forward(self, doc_embeddings, start_mappings, end_mappings, continuous_mappings, width, type_labels=None):
        start_embeddings = torch.matmul(start_mappings, doc_embeddings)
        end_embeddings = torch.matmul(end_mappings, doc_embeddings)
        start_end = torch.cat([start_embeddings, end_embeddings], dim=-1)
        continuous_embeddings = torch.matmul(continuous_mappings, doc_embeddings.unsqueeze(1))
        
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
        
        if self.with_type_embedding:
          type_labels = type_labels.view(B*S)
          type_embedding = self.type_feature(type_labels)
          vector = torch.cat((vector, type_embedding), dim=1) 

        if not isinstance(continuous_embeddings, list):
          vector = vector.view(B, S, -1)
        return vector

class GCNSpanEmbedder(nn.Module):
    def __init__(self, config, num_role_types, device):
        super(GCNSpanEmbedder, self).__init__()
        self.num_role_types = num_role_types
        self.bert_hidden_size = config.bert_hidden_size
        self.with_width_embedding = config.with_mention_width
        self.use_head_attention = config.with_head_attention
        self.with_type_embedding = config.with_type_embedding
        self.gcn_input_size = 2 * self.bert_hidden_size
        if self.use_head_attention:
            self.gcn_input_size += self.bert_hidden_size
        if self.with_width_embedding:
            self.gcn_input_size += config.embedding_dimension
        if self.with_type_embedding:
            self.gcn_input_size += config.type_embedding_dimension
        
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
        self.type_feature = nn.Embedding(200, config.type_embedding_dimension)
        self.gcn = GraphConvolution(self.gcn_input_size, self.bert_hidden_size, self.num_role_types)

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
        
    def forward(self, doc_embeddings, start_mappings, end_mappings, continuous_mappings, width, type_labels, adjm):
        start_embeddings = torch.matmul(start_mappings, doc_embeddings)
        end_embeddings = torch.matmul(end_mappings, doc_embeddings)
        start_end = torch.cat([start_embeddings, end_embeddings], dim=-1)
        continuous_embeddings = torch.matmul(continuous_mappings, doc_embeddings.unsqueeze(1))

        vector = start_end
        B = continuous_embeddings.size(0)
        S = continuous_embeddings.size(1)
        M = continuous_embeddings.size(2)
        continuous_embeddings = continuous_embeddings.view(B*S, M, -1)
        width = width.view(B*S)
        type_labels = type_labels.view(B*S)
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
        
        if self.with_type_embedding:
          type_embedding = self.type_feature(type_labels)
          vector = torch.cat((vector, type_embedding), dim=1) 

        vector = vector.view(B, S, -1)
        vector = self.gcn(vector, adjm)

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
        if config.mention_embedder == 'gcn':
            self.input_layer = config.bert_hidden_size
        else:
            self.input_layer = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2 
            if config.with_mention_width:
                self.input_layer += config.embedding_dimension
            if config.with_type_embedding:
                self.input_layer += config.type_embedding_dimension

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

    def predict_cluster(self, span_embeddings, first_idx, second_idx):
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
      first_span_embeddings = span_embeddings[first_idx]
      second_span_embeddings = span_embeddings[second_idx]

      scores = self(first_span_embeddings, second_span_embeddings)
      children = -1 * np.ones(span_num, dtype=np.int64)

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

class TransformerPairWiseClassifier(nn.Module):
  def __init__(self, config):
    super(TransformerPairWiseClassifier, self).__init__()  

    self.input_layer = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2 
    if config.with_mention_width:
        self.input_layer += config.embedding_dimension
    if config.get('with_type_embedding', False):
        self.input_layer += config.type_embedding_dimension

    if config.get('num_encoder_layers', 1) > 0:
      self.transformer = torch.nn.Transformer(d_model=self.input_layer, 
                                              nhead=1, 
                                              num_encoder_layers=1, 
                                              num_decoder_layers=1)
    else:
      transformer_decoder = nn.TransformerDecoderLayer(d_model=self.input_layer,
                                                       nheads=1)
      self.transformer = torch.nn.TransformerDecoder(transformer_decoder,
                                                     num_decoder_layers=1)

  def forward(self, first, second):
    '''
    :param first: FloatTensor of size (num. of span pairs, span embed dim)
    :param second: FloatTensor of size (num. of span pairs, span embed dim)
    :return score: FloatTensor of size (num. of span pairs, 1) 
    '''
    first = first.unsqueeze(0)
    second = second.unsqueeze(0)

    d = first.size(-1)
    first_c = self.transformer(second, second).squeeze(0)
    # scale =  torch.tensor(d ** 0.5, dtype=torch.float, device=first.device)
    scores = torch.sum(first * first_c, dim=-1).t() # / scale
    return scores

  def predict_cluster(self, span_embeddings, first_idxs, second_idxs):
    span_num = span_embeddings.size(0)

    # Compute pairwise scores for mention pairs specified
    scores = self(span_embeddings[first_idxs], span_embeddings[second_idxs])
    
    # Compute the pairwise score matrix
    row_idxs = [i for i in range(span_num) for j in range(span_num)]
    col_idxs = [j for i in range(span_num) for j in range(span_num)]
    S = self(span_embeddings[row_idxs], span_embeddings[col_idxs])
    S = S.view(span_num, span_num).cpu().detach().numpy()

    # Compute the adjacency matrix
    A = np.zeros((span_num, span_num), dtype=np.float)
    for row_idx in row_idxs:
      col_idx = np.argmax(S[row_idx]) 
      A[row_idx, col_idx] = 1.
      A[col_idx, row_idx] = 1.
  
    # Find the connected components of the graph
    clusters = find_connected_components(A)
    clusters = {k:c for k, c in enumerate(clusters)}
    # print('Number of clusters: ', len(clusters))
    return clusters, scores

class StarTransformerClassifier(nn.Module):
  def __init__(self, config):
    super(StarTransformerClassifier, self).__init__()  

    self.input_layer = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2 
    if config.with_mention_width:
        self.input_layer += config.embedding_dimension
    if config.get('with_type_embedding', False):
        self.input_layer += config.type_embedding_dimension

    if config.get('num_encoder_layers', 1) > 0:
      self.transformer = torch.nn.Transformer(d_model=self.input_layer, 
                                              nhead=1, 
                                              num_encoder_layers=1, 
                                              num_decoder_layers=1)
    else:
      transformer_decoder = nn.TransformerDecoderLayer(d_model=self.input_layer,
                                                       nheads=1)
      self.transformer = torch.nn.TransformerDecoder(transformer_decoder,
                                                     num_decoder_layers=1)

  def forward(self, x, center_map, neighbor_map, 
              first_idxs=None, second_idxs=None): # TODO Add mask
    '''
    :param x: FloatTensor of size (sequence length, embed dim)
    :param neighbors: FloatTensor of size (sequence length, num. of neighbors, embed dim)
    :return scores: FloatTensor of size (num. pairs,) 
    '''
    embs = self.transformer(x, x).squeeze(0)
    c = torch.matmul(center_map, x)
    neighors = torch.matmul(neighbor_map, x)
    embs_c = torch.matmul(center_map, embs)
    embs_neighbors = torch.matmul(neighbor_map, embs)
    n = embs_c.size(0)
    n_neighbors = embs_neighbors.size(1)
     
    if not first_idxs or not second_idxs:
      first_idxs, second_idxs = zip(*list(combinations(range(n), 2)))
    first_arg_idxs, second_arg_idxs = zip(*list((i, j) for i in range(n_neighbors) for j in range(n_neighbors)))
    scores_c = self.pairwise_score(c[first_idxs], embs_c[second_idxs])
    scores_neighbor = []
    for first_idx, second_idx in zip(first_idxs, second_idxs):
      scores_neighbor.append(self.alignment_score(
                          self.pairwise_score(neighbors[first_idx, first_arg_idxs],
                                              embs_neighbors[second_idx, second_arg_idxs]).view(n_neighbors, n_neighbors))
                          )
    scores_neighbors = torch.stack(scores_neighbors)
    return scores + scores_neighbors

  def pairwise_score(self, first, second):
    return torch.sum(first * second, dim=-1)

  def alignment_score(self, score_mat, dim=-1):
    return score_mat.max(-1)[0].mean()
    
  def predict_cluster(self, x, center_map, neighbor_map, first_idxs, second_idxs):
    span_num = x.size(0)

    # Compute pairwise scores for mention pairs specified
    scores = self(x, center_map, neighbor_map, first_idxs, second_idxs)
    
    # Compute the pairwise score matrix
    row_idxs = [i for i in range(span_num) for j in range(span_num)]
    col_idxs = [j for i in range(span_num) for j in range(span_num)]
    S = self(x, neighbors, row_idxs, col_idxs)
    S = S.view(span_num, span_num).cpu().detach().numpy()

    # Compute the adjacency matrix
    A = np.zeros((span_num, span_num), dtype=np.float)
    for row_idx in row_idxs:
      col_idx = np.argmax(S[row_idx]) 
      A[row_idx, col_idx] = 1.
      A[col_idx, row_idx] = 1.
  
    # Find the connected components of the graph
    clusters = find_connected_components(A)
    clusters = {k:c for k, c in enumerate(clusters)}
    # print('Number of clusters: ', len(clusters))
    return clusters, scores
    

def find_connected_components(A):
  def _dfs(v, c):
    for u in range(n):
      if (A[v, u] or A[u, v]) and visited[u] < 0:
        visited[u] = 1
        c.append(u)
        c = _dfs(u, c)
    return c

  n = A.shape[0]
  visited = -1 * np.ones(n)
  v = 0
  components = []
  for v in range(n):
    if visited[v] < 0:
      c = _dfs(v, [])
      components.append(c)
  return components  

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
