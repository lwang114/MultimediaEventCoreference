import torch
import torch.nn as nn
import torch.nn.functional as F
from text_models import BiLSTM

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)

class VisCorefAttender(nn.Module):
  def __init__(self,
               input_dim,
               hidden_dim,
               n_class):
    super(VisCorefAttender, self).__init__()
    self.visual_embedder = nn.Sequential(
                             nn.Linear(n_class, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, hidden_dim)
                           )
    self.mention_embedder = nn.Sequential(
                              nn.Linear(input_dim, hidden_dim),
                              nn.ReLU(),
                              nn.Linear(hidden_dim, hidden_dim)
                            )
    self.attention = nn.Sequential(
                       nn.Linear(2 * hidden_dim, hidden_dim),
                       nn.ReLU(),
                       nn.Linear(hidden_dim, 1)
                     )

  def forward(self, mention_embeddings, class_labels, 
              mention_mask, class_mask):
    """
    Args :
      mention_embeddings : FloatTensor of size (batch size, max mention num, input size)
      class_labels : LongTensor of size (batch size, max class num, n class)
      mention_mask : FloatTensor of size (batch size, max mention num) 
      class_mask : FloatTensor of size (batch size, max class num)
    
    Returns : 
      attn_weights : FloatTensor of size (batch size, max mention num, max class num)
    """
    device = mention_embeddings.device
    visual_embeddings = self.visual_embedder(class_labels)
    mention_embeddings = self.mention_embedder(mention_embeddings)
    attn_weights = self.attention(visual_embeddings.unsqueeze(-3) 
                                  * mention_embeddings.unsqueeze(-2))
    attn_weights = attn_weights * mention_mask.unsqueeze(-1)\
                   * class_mask.unsqueeze(-2)
    attn_weights = torch.where(attn_weights != 0,
                               attn_weights,
                               torch.tensor(-1e10, device=device))
    return attn_weights


class ClassAttender(nn.Module):
  def __init__(self,
               input_dim,
               hidden_dim,
               n_class):
    super(ClassAttender, self).__init__()
    self.attention = nn.Linear(input_dim, n_class, bias=False)
    self.classifier = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1)
                      )

  def forward(self, x, mask):
    """
    Args :
      x : FloatTensor of size (batch size, seq length, input size)
      mask : FloatTensor of size (batch size, seq length)
    
    Returns :
      out : FloatTensor of size (batch size, n class)
      attn_weights = FloatTensor of size (batch size, n class, seq len)
    """
    attn_weights = self.attention(x).permute(0, 2, 1)
    attn_weights = attn_weights * mask.unsqueeze(-2)
    attn_weights = torch.where(attn_weights != 0,
                               attn_weights,
                               torch.tensor(-1e10, device=x.device))
    
    # (batch size, n class, seq length)
    attn_weights = F.softmax(attn_weights, dim=-1)
    # (batch size, n class, input size)
    attn_applied = torch.bmm(attn_weights, x)
    # (batch size, n class)
    out = self.classifier(attn_applied).squeeze(-1)
    return out, attn_weights


class BiLSTMVideoEncoder(nn.Module):
  def __init__(self,
               input_dim,
               embedding_dim,
               num_layers=1):
    super(BiLSTMVideoEncoder, self).__init__()
    self.encoder = BiLSTM(input_dim,
                          embedding_dim,
                          num_layers=num_layers)
    
  def forward(self, x, mask): # TODO Check this
    """ 
    Args : 
      x : FloatTensor of size (batch size, num. of regions, num. of frames per region, input dim)
      mask : FloatTensor of size (batch size, num. of regions, num. of frames per region)

    Returns : FloatTensor of size (batch size, num. of regions, embedding dim)
    """
    device = x.device
    B, N, F, D = x.size() 
    action_output = self.encoder(x.view(B*N, F, -1))
    action_len = mask.sum(-1).unsqueeze(-1)
    action_output = (mask.unsqueeze(-1) * action_output.view(B, N, F, -1)).sum(-2)\
                    / torch.max(action_len, torch.ones(1, device=device))
    return action_output


class CrossmediaPairWiseClassifier(nn.Module):
  def __init__(self, config):
    super(CrossmediaPairWiseClassifier, self).__init__()
    self.input_layer = 3 * config.hidden_layer 
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

    self.crossmedia_pairwise_mlp = nn.Sequential(
                                      nn.Linear(4, 4),
                                      nn.ReLU(),
                                      nn.Linear(4, 1)
                                   ) 
    self.crossmedia_pairwise_mlp.apply(init_weights)

  def forward(self, first, second):
    """
    Args :
        first : FloatTensor of size (num. of pairs, common space dim.)
        second : FloatTensor of size (num. of pairs, common space dim.)
    
    Returns :
        scores : FloatTensor of size (num. of pairs, 1)
    """ 
    # XXX return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1))   
    return torch.sum(first * second, dim=-1, keepdim=True) 

  def crossmedia_score(self, first_idxs, second_idxs, attention_map, use_null=False):
    """
    Compute similarity between mentions based on their crossmedia scores
    
    Args :
        first_idxs : FloatTensor of size (batch size, num. of pairs)
        second_idxs : FloatTensor of size (batch size, num. of pairs)
        attention_map : FloatTensor of size (batch size, num. of events, num. of actions) 
    
    Returns :
        scores : FloatTensor of size (batch size, num. of pairs, 1)
    """
    first_attention = attention_map[first_idxs]
    second_attention = attention_map[second_idxs]
    first_attention = F.softmax(first_attention, dim=-1)
    second_attention = F.softmax(second_attention, dim=-1)
    if use_null:
      first_attention = first_attention[:, 1:]
      second_attention = second_attention[:, 1:] 
    first_score = first_attention.max(-1)[0].unsqueeze(-1)
    second_score = second_attention.max(-1)[0].unsqueeze(-1)
    pw_score = (first_attention * second_attention).max(-1)[0].unsqueeze(-1)
    return self.crossmedia_pairwise_mlp(
              torch.cat((first_score, second_score, first_score * second_score, pw_score), dim=1)) 
