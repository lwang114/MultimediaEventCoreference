import numpy as np
from sklearn.cluster import KMeans
from scipy.special import logsumexp
from copy import deepcopy

class RegionVGMM(object):
  """
  A Region Visual Gaussian mixture model (VGMM)/
  Parameters
  ----------
  K : int
      The number of mixture components
  """
  def __init__(self, X, K, assignments='kmeans', var=1., lr=0.1, vec_ids=None):
    self.K_max = K
    self.D = X.shape[-1]
    self.means = np.zeros((K, self.D))
    self.lr = lr
    self.X = X
    self.var = var
    self.vec_ids = vec_ids # TODO Make this nonempty by default 
    self.setup_components()    

  def setup_components(self, assignments='kmeans'): 
    if isinstance(assignments, str) and assignments == 'kmeans':
      keep = [i_ex*75+i_region for i_ex in range(int(self.X.shape[0]/75)) for i_region in range(15)] # Only select a subset of examples for KMeans clustering
      keep = np.asarray(keep)
      self.means = KMeans(n_clusters=self.K_max).fit(self.X[keep]).cluster_centers_ 
    else:
      raise NotImplementedError
 
  def log_prob_z(self, i, normalize=True):
    """
    Parameters
    ----------
    i : int
        index of the image feature vector sequence
    
    Returns
    -------
    log_prob_z : length K vector 
             [[p(z_i=k|y_i) for k in range(K)] for i in range(L)]
    """
    log_prob_zs = []
    for j in self.vec_ids[i]:
      log_prob_z = self.log_prob_z_given_X(self.X[j], normalize=normalize) 
      log_prob_zs.append(log_prob_z)
    return np.asarray(log_prob_zs)

  def log_prob_z_given_X(self, X, normalize=True):
    log_prob_z = - np.sum((X - self.means) ** 2, axis=1) / self.var
    if normalize:
      log_prob_z -= logsumexp(log_prob_z)
    return log_prob_z

  def log_post_pred(self, i):
    """
    Parameters
    ----------
    i : int
        index of the image feature vector sequence
    
    Returns
    -------
    log_post_pred : length K vector
             [log (1/L \sum_{j=1}^L p(z_j^i=k|y_j^i)) for k in range(K)]
    """
    L = len(self.vec_ids[i])
    log_post_preds = []
    for j in self.vec_ids[i]:    
      log_post_pred_j = - np.sum((self.X[j] - self.means) ** 2, axis=1) / self.var
      log_post_pred_j -= logsumexp(log_post_pred_j)
      log_post_preds.append(log_post_pred_j)
   
    return logsumexp(np.asarray(log_post_preds), axis=0) - np.log(L)

  def update_components(self, indices, ws):
    assert len(indices) == ws.shape[0]
    vs = self.X[indices]
    self.means += self.lr * np.dot(np.transpose(ws), vs)
   
  def swap_clusters(self, k1, k2):
    tmp = deepcopy(self.means[k2])
    self.means[k2] = self.means[k1]
