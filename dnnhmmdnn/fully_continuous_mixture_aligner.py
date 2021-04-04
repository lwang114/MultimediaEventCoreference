#-----------------------------------------------------------------------------------# 
#                           CONTINUOUS MIXTURE ALIGNER CLASS                        #
#-----------------------------------------------------------------------------------# 
import numpy as np
import logging
import os
import json
import codecs
import torch
import nltk
from nltk.stem import WordNetLemmatizer
import pyhocon
from region_vgmm import *
from negative_square import NegativeSquare

logger = logging.getLogger(__name__)
EPS = 1e-15
class FullyContinuousMixtureAligner(object):
  """An alignment model based on Brown et. al., 1993. capable of modeling continuous bilingual sentences"""
  def __init__(self, source_features_train, target_features_train, configs):
    self.Ks = configs.get('n_src_vocab', 80)
    self.Kt = configs.get('n_trg_vocab', 2001)
    self.use_null = configs.get('use_null', True)
    self.pretrained_vgmm_model = configs.get('pretrained_vgmm_model', None)
    self.pretrained_translateprob = configs.get('pretrained_translateprob', None)
    var = configs.get('var', 160.) # XXX
    logger.info('n_src_vocab={}, n_trg_vocab={}'.format(self.Ks, self.Kt))
    self.alpha = configs.get('alpha', 0.)
    if target_features_train[0].ndim <= 1:
      self.trg_embedding_dim = 1 
    else:
      self.trg_embedding_dim = target_features_train[0].shape[-1]
    print('target embedding dimension={}'.format(self.trg_embedding_dim))

    self.src_vec_ids_train = []
    start_index = 0
    for ex, src_feat in enumerate(source_features_train):
      if self.use_null:        
        if self.trg_embedding_dim == 1:
          target_features_train[ex] = [self.Kt-1]+target_features_train[ex]
          # print('Warning: not supporting NULL symbol for continuous input yet')
      src_vec_ids = []
      for t in range(len(src_feat)):
        src_vec_ids.append(start_index+t)
      start_index += len(src_feat)
      self.src_vec_ids_train.append(src_vec_ids)
    
    self.src_model = RegionVGMM(np.concatenate(source_features_train, axis=0),
                                self.Ks,
                                var=var,
                                vec_ids=self.src_vec_ids_train,
                                pretrained_model=self.pretrained_vgmm_model)
    self.src_feats = self.src_model.X
    self.trg_feats = target_features_train
    if self.pretrained_translateprob:
      self.P_ts = np.load(self.pretrained_translateprob)
      print('Loaded pretrained translation probabilities')
    else:
      self.P_ts = 1./self.Ks * np.ones((self.Kt, self.Ks))
    self.trg2src_counts = np.zeros((self.Kt, self.Ks))

  def compute_forward_probs(self, trg_sent, src_sent): # TODO Reverse the definition
    L = src_sent.shape[0]
    T = trg_sent.shape[0]
    A = np.ones((L, L)) / max(L, 1)
    init = np.ones(L) / max(L, 1)
    forward_probs = np.zeros((T, L, self.Kt))
    scales = np.zeros((T,))
    
    probs_x_t_given_z = trg_sent @ self.P_ts.T
    forward_probs[0] = np.tile(init[:, np.newaxis], (1, self.Kt)) * src_sent * probs_x_t_given_z[0] 
    scales[0] = np.sum(forward_probs[0])
    forward_probs[0] /= np.maximum(scales[0], EPS)
    A_diag = np.diag(np.diag(A))
    A_offdiag = A - A_diag
    for t in range(T-1):
      probs_x_t_z_given_y = src_sent * probs_x_t_given_z[t+1]
      forward_probs[t+1] += (A_diag @ forward_probs[t]) * probs_x_t_given_z[t+1]
      forward_probs[t+1] += ((A_offdiag.T @ np.sum(forward_probs[t], axis=-1)) * probs_x_t_z_given_y.T).T
      scales[t+1] = np.sum(forward_probs[t+1])
      forward_probs[t+1] /= max(scales[t+1], EPS)
    return forward_probs, scales
      
  def compute_backward_probs(self, trg_sent, src_sent, scales):
    T = trg_sent.shape[0]
    L = src_sent.shape[0]
    A = np.ones((L, L)) / max(L, 1)
    init = np.ones(L) / max(L, 1)
    backward_probs = np.zeros((T, L, self.Kt))
    backward_probs[T-1] = 1. / max(scales[T-1], EPS)

    A_diag = np.diag(np.diag(A))
    A_offdiag = A - A_diag
    probs_x_t_given_z = trg_sent @ self.P_ts.T
    for t in range(T-1, 0, -1):
      probs_x_t_z_given_y = src_sent * probs_x_t_given_z[t]
      backward_probs[t-1] = A_diag @ (backward_probs[t] * probs_x_t_given_z[t])
      backward_probs[t-1] += np.tile(A_offdiag @ np.sum(backward_probs[t] * probs_x_t_z_given_y, axis=-1)[:, np.newaxis], (1, self.Kt))
      backward_probs[t-1] /= max(scales[t-1], EPS)
    return backward_probs
    
  def update_counts(self):
    # Update alignment counts
    log_probs = []
    self.trg2src_counts[:] = 0.
    for i, (trg_feat, src_vec_ids) in enumerate(zip(self.trg_feats, self.src_vec_ids_train)):
      src_feat = self.src_feats[src_vec_ids]
      C_ts, log_prob_i = self.update_counts_i(i, src_feat, trg_feat)
      self.trg2src_counts += C_ts
      log_probs.append(log_prob_i)

    self.P_ts = deepcopy(self.translate_prob())
    return np.mean(log_probs)

  def update_counts_i(self, i, src_feat, trg_feat):
    src_sent = np.exp(self.src_model.log_prob_z(i, normalize=False))
    trg_sent = trg_feat

    V_src = to_one_hot(src_sent, self.Ks)
    V_trg = to_one_hot(trg_sent, self.Kt)
   
    forward_probs, scales = self.compute_forward_probs(V_src, V_trg)
    backward_probs = self.compute_backward_probs(V_src, V_trg, scales)
    norm_factor = np.sum(forward_probs * backward_probs, axis=(1, 2), keepdims=True) 
    new_state_counts = forward_probs * backward_probs / np.maximum(norm_factor, EPS) 
    C_ts = np.sum(new_state_counts, axis=1).T @ (V_src / np.maximum(np.sum(V_src, axis=1, keepdims=True), EPS))
    log_prob = np.log(np.maximum(scales, EPS)).sum()
    return C_ts, log_prob

  def update_components(self):
    means_new = np.zeros(self.src_model.means.shape)
    counts = np.zeros((self.Ks,))
    for i, (trg_feat, src_feat) in enumerate(zip(self.trg_feats, self.src_feats)):
      if len(trg_feat) == 0 or len(self.src_feats[i]) == 0:
        continue 
      trg_sent = trg_feat
      prob_f_given_y = self.prob_s_given_tsent(trg_sent)
      prob_f_given_x = np.exp(self.src_model.log_prob_z(i))
      post_f = prob_f_given_y * prob_f_given_x
      post_f /= np.maximum(np.sum(post_f, axis=1, keepdims=True), EPS)
  
      # Update target word counts of the target model
      indices = self.src_vec_ids_train[i]
     
      means_new += np.sum(post_f[:, :, np.newaxis] * self.src_model.X[indices, np.newaxis], axis=0)
      counts += np.sum(post_f, axis=0)
      # self.update_components_exact(i, ws=post_f, method='exact') 
    self.src_model.means = deepcopy(means_new / np.maximum(counts[:, np.newaxis], EPS)) 
     
  def trainEM(self, n_iter, out_file):
    for i_iter in range(n_iter):
      log_prob = self.update_counts()
      self.update_components() # XXX
      print('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      logger.info('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      if (i_iter + 1) % 5 == 0:
        with open('{}_{}_means.json'.format(out_file, i_iter), 'w') as fm,\
             open('{}_{}_transprob.json'.format(out_file, i_iter), 'w') as ft:
          json.dump(self.src_model.means.tolist(), fm, indent=4, sort_keys=True)
          json.dump(self.P_ts.tolist(), ft, indent=4, sort_keys=True)
          
        np.save('{}_{}_means.npy'.format(out_file, i_iter), self.src_model.means)
        np.save('{}_{}_transprob.npy'.format(out_file, i_iter), self.P_ts)

  def translate_prob(self):
    return (self.alpha / self.Ks + self.trg2src_counts) / np.maximum(self.alpha + np.sum(self.trg2src_counts, axis=-1, keepdims=True), EPS)
  
  def prob_s_given_tsent(self, trg_sent):
    V_trg = to_one_hot(trg_sent, self.Kt)
    return np.mean(V_trg @ self.P_ts, axis=0) 
    
  def align_sents(self, source_feats_test, target_feats_test, score_type='max'): 
    alignments = []
    scores = []
    for src_feat, trg_feat in zip(source_feats_test, target_feats_test):
      trg_sent = trg_feat
      src_sent = [np.exp(self.src_model.log_prob_z_given_X(src_feat[i])) for i in range(len(src_feat))]
      V_trg = to_one_hot(trg_sent, self.Kt)
      V_src = to_one_hot(src_sent, self.Ks)
      P_a = V_trg @ self.P_ts @ V_src.T
      if score_type == 'max':
        scores.append(np.prod(np.max(P_a, axis=0)))
      elif score_type == 'mean':
        scores.append(np.prod(np.mean(P_a, axis=0)))
      else:
        raise ValueError('Score type not implemented')
      alignments.append(np.argmax(P_a, axis=0)) 
    return alignments, np.asarray(scores)

  def retrieve(self, source_features_test, target_features_test, out_file, kbest=10):
    n = len(source_features_test)
    print(n)
    scores = np.zeros((n, n))
    for i_utt in range(n):
      if self.use_null:
        src_feats = [source_features_test[i_utt] for _ in range(n)] 
        trg_feats = [[self.Kt - 1] + target_features_test[j_utt] for j_utt in range(n)]
      else:
        src_feats = [source_features_test[i_utt] for _ in range(n)] 
        trg_feats = [target_features_test[j_utt] for j_utt in range(n)]
       
      _, scores[i_utt] = self.align_sents(src_feats, trg_feats, score_type='max') 

    I_kbest = np.argsort(-scores, axis=1)[:, :kbest]
    P_kbest = np.argsort(-scores, axis=0)[:kbest]
    n = len(scores)
    I_recall_at_1 = 0.
    I_recall_at_5 = 0.
    I_recall_at_10 = 0.
    P_recall_at_1 = 0.
    P_recall_at_5 = 0.
    P_recall_at_10 = 0.

    for i in range(n):
      if I_kbest[i][0] == i:
        I_recall_at_1 += 1
      
      for j in I_kbest[i][:5]:
        if i == j:
          I_recall_at_5 += 1
       
      for j in I_kbest[i][:10]:
        if i == j:
          I_recall_at_10 += 1
      
      if P_kbest[0][i] == i:
        P_recall_at_1 += 1
      
      for j in P_kbest[:5, i]:
        if i == j:
          P_recall_at_5 += 1
       
      for j in P_kbest[:10, i]:
        if i == j:
          P_recall_at_10 += 1

    I_recall_at_1 /= n
    I_recall_at_5 /= n
    I_recall_at_10 /= n
    P_recall_at_1 /= n
    P_recall_at_5 /= n
    P_recall_at_10 /= n
     
    print('Image Search Recall@1: ', I_recall_at_1)
    print('Image Search Recall@5: ', I_recall_at_5)
    print('Image Search Recall@10: ', I_recall_at_10)
    print('Captioning Recall@1: ', P_recall_at_1)
    print('Captioning Recall@5: ', P_recall_at_5)
    print('Captioning Recall@10: ', P_recall_at_10)
    logger.info('Image Search Recall@1, 5, 10: {}, {}, {}'.format(I_recall_at_1, I_recall_at_5, I_recall_at_10))
    logger.info('Captioning Recall@1, 5, 10: {}, {}, {}'.format(P_recall_at_1, P_recall_at_5, P_recall_at_10))

    fp1 = open(out_file + '_image_search.txt', 'w')
    fp2 = open(out_file + '_image_search.txt.readable', 'w')
    for i in range(n):
      I_kbest_str = ' '.join([str(idx) for idx in I_kbest[i]])
      fp1.write(I_kbest_str + '\n')
    fp1.close()
    fp2.close() 

    fp1 = open(out_file + '_captioning.txt', 'w')
    fp2 = open(out_file + '_captioning.txt.readable', 'w')
    for i in range(n):
      P_kbest_str = ' '.join([str(idx) for idx in P_kbest[:, i]])
      fp1.write(P_kbest_str + '\n\n')
      fp2.write(P_kbest_str + '\n\n')
    fp1.close()
    fp2.close()  

  def move_counts(self, k1, k2):
    self.trg2src_counts[:, k2] = self.trg2src_counts[:, k1]
    self.trg2src_counts[:, k1] = 0.

  def print_alignment(self, out_file):
    align_dicts = []
    for i, (src_vec_ids, trg_feat) in enumerate(zip(self.src_vec_ids_train, self.trg_feats)):
      src_feat = self.src_feats[src_vec_ids]
      alignment = self.align_sents([src_feat], [trg_feat])[0][0]
      src_sent = np.argmax(self.src_model.log_prob_z(i), axis=1)
      align_dicts.append({'alignment': alignment.tolist(),
                          'image_concepts': src_sent.tolist()})
    with open(out_file, 'w') as f:
      json.dump(align_dicts, f, indent=4, sort_keys=True)
  
def to_one_hot(sent, K):
  sent = np.asarray(sent)
  if len(sent.shape) < 2:
    es = np.eye(K)
    sent = np.asarray([es[int(w)] if w < K else 1./K*np.ones(K) for w in sent])
    return sent
  else:
    return sent

def load_data(config):
  """
  Returns:
      src_feats_train: a list of arrays of shape (src sent length, src dimension)
      trg_feats_train: a list of arrays of shape (trg sent length, trg dimension)
      src_feats_test: a list of arrays of shape (src sent length, src dimension)
      trg_feats_test: a list of arrays of shape (trg sent length, trg dimension)
  """
  lemmatizer = WordNetLemmatizer() 
  event_mentions_train = json.load(codecs.open(os.path.join(config['data_folder'], 'train_events.json'), 'r', 'utf-8'))
  event_mentions_test = json.load(codecs.open(os.path.join(config['data_folder'], 'test_events.json'), 'r', 'utf-8'))
  visual_feats = np.load(os.path.join(config['data_folder'], 'train_mmaction_event_feat.npz'))
  doc_to_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in visual_feats}

  vocab = dict()
  vocab_freq = {}
  label_dict_train = {}
  label_dict_test = {}
  for m in event_mentions_train + event_mentions_test:
    trigger = m['tokens']
    trigger = lemmatizer.lemmatize(trigger.lower(), pos='v')
    if not trigger in vocab:
      vocab[trigger] = len(vocab)
      vocab_freq[trigger] = 1
    else:
      vocab_freq[trigger] += 1
  json.dump(vocab_freq, open('vocab_freq.json', 'w'), indent=2)
  vocab_size = len(vocab)

  for m in event_mentions_train:
    if m['doc_id'] in doc_to_feat:
      if not m['doc_id'] in label_dict_train:
        label_dict_train[m['doc_id']] = {}
      token = lemmatizer.lemmatize(m['tokens'].lower(), pos='v')
      label_dict_train[m['doc_id']][(min(m['tokens_ids']), max(m['tokens_ids']))] = vocab[token] 

  for m in event_mentions_test:
    if m['doc_id'] in doc_to_feat:
      if not m['doc_id'] in label_dict_test:
        label_dict_test[m['doc_id']] = {}
      token = lemmatizer.lemmatize(m['tokens'].lower(), pos='v')
      label_dict_test[m['doc_id']][(min(m['tokens_ids']), max(m['tokens_ids']))] = vocab[token]
  print(f'Vocab size: {vocab_size}')
  print(f'Number of training examples: {len(label_dict_train)}')
  print(f'Number of test examples: {len(label_dict_test)}')

  src_feats_train = []
  trg_feats_train = []
  src_feats_test = []
  trg_feats_test = []
  for feat_idx, doc_id in enumerate(sorted(label_dict_train)): # XXX
    feat_id = doc_to_feat[doc_id]
    src_feats_train.append(visual_feats[feat_id])
    trg_sent = [label_dict_train[doc_id][span] for span in sorted(label_dict_train[doc_id])]
    trg_feats_train.append(to_one_hot(trg_sent, vocab_size))

  for feat_idx, doc_id in enumerate(sorted(label_dict_test)): 
    feat_id = doc_to_feat[doc_id]
    src_feats_test.append(visual_feats[feat_id])
    trg_sent = [label_dict_test[doc_id][span] for span in sorted(label_dict_test[doc_id])]
    trg_feats_test.append(to_one_hot(trg_sent, vocab_size))

  return src_feats_train, trg_feats_train, src_feats_test, trg_feats_test, vocab

if __name__ == '__main__':
  config_file = '../configs/config_dnnhmmdnn_video_m2e2.json'
  config = pyhocon.ConfigFactory.parse_file(config_file)
  Ks = 33

  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  logging.basicConfig(filename=os.path.join(config['model_path'], 'train.log'))
  
  src_feats_train, trg_feats_train, src_feats_test, trg_feats_test, vocab = load_data(config)
  Kt = len(vocab)
  aligner = FullyContinuousMixtureAligner(src_feats_train, trg_feats_train, configs={'n_trg_vocab':Kt, 'n_src_vocab':Ks})
  aligner.trainEM(15, os.path.join(config['model_path'], 'mixture'))
  aligner.print_alignment(os.path.join(config['model_path'], 'alignment.json'))
  aligner.retrieve(src_feats_test, trg_feats_test, os.path.join(config['model_path'], 'retrieval'))
