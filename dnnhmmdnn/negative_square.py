import torch
import torch.nn as nn

class NegativeSquare(nn.Module):
  def __init__(self, codebook, precision):
    super(NegativeSquare, self).__init__()
    self.codebook = nn.Parameter(codebook, requires_grad=codebook.requires_grad)
    self.precision = nn.Parameter(precision * torch.ones((1,)), requires_grad=False) # TODO Make this trainable                                                                                            

  def forward(self, x, compute_softmax=False):
    """                                                                                                                                                                                                    
    Args:                                                                                                                                                                                                  
        x: B x T x D array of acoustic features                                                                                                                                                                                                                                                                                                                                                                       
    Returns:                                                                                                                                                                                                       score: B x T x K array of gaussian log posterior probabilities                                                                                                                                     
             [[[precision*(x[t] - mu[c[t]])**2 for k in range(K)] for t in range(T)] for b in range(B)]                                                                                                    
    """
    # score1 = -(x.unsqueeze(-2) - self.codebook).pow(2).sum(-1)
    # score1 = score1 * self.precision
    # return self.precision * score
    B = x.size(0)
    T = x.size(1)
    score = torch.stack([torch.stack([-self.precision * torch.sum((x[b, t] - self.codebook)**2, dim=1) for t in range(T)]) for b in range(B)])
    if compute_softmax:
      score = score.softmax(-1)
    return score
