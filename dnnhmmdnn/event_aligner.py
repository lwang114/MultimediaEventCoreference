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
import argparse
import itertools
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from region_vgmm import *
from negative_square import NegativeSquare
from evaluator import Evaluation, CoNLLEvaluation

logger = logging.getLogger(__name__)
EPS = 1e-100
np.random.seed(2)
class FullyContinuousMixtureAligner(object):
  """An alignment model based on Brown et. al., 1993. capable of modeling continuous bilingual sentences"""
  def __init__(self, source_features_train, target_features_train, configs):
    self.Ks = configs.get('Ks', 50)
    self.Kt = configs.get('Kt', target_features_train[0].shape[-1])
    self.use_null = configs.get('use_null', False)
    self.pretrained_vgmm_model = configs.get('pretrained_vgmm_model', None)
    self.pretrained_translateprob = configs.get('pretrained_translateprob', None)
    self.vocab = configs.get('vocab', None)
    var = configs.get('var', 30.) # XXX
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
          target_features_train[ex] = target_features_train[ex] + [self.Kt-1]

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
    self.src_null_vec = self.src_model.X.mean(axis=0)
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
      self.update_components()
      print('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      logger.info('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      if (i_iter + 1) % 5 == 0:
        with open('{}_{}_means.json'.format(out_file, i_iter), 'w') as fm,\
             open('{}_{}_transprob.json'.format(out_file, i_iter), 'w') as ft:
          json.dump(self.src_model.means.tolist(), fm, indent=4, sort_keys=True)
          if self.vocab is not None:
            P_ts_dict = dict()
            for v, p in zip(self.vocab, self.P_ts.tolist()):
              P_ts_dict[v] = p
            json.dump(P_ts_dict, ft, indent=2, sort_keys=True) 
          else:
            json.dump(self.P_ts.tolist(), ft, indent=2, sort_keys=True)

        np.save('{}_{}_means.npy'.format(out_file, i_iter), self.src_model.means)
        np.save('{}_{}_transprob.npy'.format(out_file, i_iter), self.P_ts)

  def translate_prob(self):
    return (self.alpha / self.Ks + self.trg2src_counts) / np.maximum(self.alpha + np.sum(self.trg2src_counts, axis=-1, keepdims=True), EPS)
  
  def prob_s_given_tsent(self, trg_sent):
    V_trg = to_one_hot(trg_sent, self.Kt)
    return np.mean(V_trg @ self.P_ts, axis=0) 
    
  def align_sents(self, source_feats_test, 
                  target_feats_test, 
                  alignment_type='text'): 
    alignments = []
    scores = []
    P_a_norm = []
    for src_feat, trg_feat in zip(source_feats_test, target_feats_test):
      trg_sent = trg_feat
      if alignment_type.split('_')[0] == 'text':
        src_feat = np.concatenate([self.src_null_vec[np.newaxis], src_feat], axis=0)
      src_sent = [np.exp(self.src_model.log_prob_z_given_X(src_feat[i], normalize=False))\
                  for i in range(len(src_feat))]

      V_trg = to_one_hot(trg_sent, self.Kt)
      V_src = to_one_hot(src_sent, self.Ks)

      trg_null_prob = V_src.T.mean(axis=0, keepdims=True)
      P_a = V_trg @ self.P_ts @ V_src.T

      if alignment_type.split('_')[0] == 'text':
        P_a = np.concatenate([trg_null_prob, P_a], axis=0)
        scores.append(np.prod(np.max(P_a, axis=0)))
        alignments.append(np.argmax(P_a, axis=1)) 
      elif alignment_type == 'image':
        scores.append(np.prod(np.max(P_a, axis=0)))
        alignments.append(np.argmax(P_a, axis=0))
      else:
        raise ValueError('Alignment type not implemented')

      P_a_norm.append(P_a / P_a.sum(axis=-1, keepdims=True))
    return alignments, np.asarray(scores), P_a_norm

  def align(self, 
            source_feats,
            target_feats,
            source_labels,
            target_labels,
            alignment_type='image',
            out_prefix='align'):
    src_vocab = dict()
    trg_vocab = dict()
    for src_label in source_labels:
      for y in src_label:
        if not y in src_vocab:
          src_vocab[y] = len(src_vocab)

    for trg_label in target_labels:
      for y in trg_label:
        if not y in trg_vocab:
          trg_vocab[y] = len(trg_vocab)

    n_src_vocab = len(src_vocab)
    n_trg_vocab = len(trg_vocab)
    alignments, _, P_a_norm = self.align_sents(source_feats,
                                               target_feats,
                                               alignment_type=alignment_type) 

    if alignment_type == 'image':
      confusion = np.zeros((n_src_vocab, n_trg_vocab))
    elif alignment_type == 'text':
      src_vocab['###NULL###'] = len(src_vocab)
      n_src_vocab += 1
      confusion = np.zeros((n_src_vocab, n_trg_vocab))

    for src_label, trg_label, alignment in zip(source_labels,
                                               target_labels,
                                               alignments):
      if alignment_type == 'image':
        for src_idx, y in enumerate(src_label):
          y_pred = trg_label[alignment[src_idx]]
          confusion[src_vocab[y], trg_vocab[y_pred]] += 1
      elif alignment_type == 'text':
        for trg_idx, y in enumerate(trg_label):
          y_pred = src_label[alignment[trg_idx+1]-1]
          confusion[src_vocab[y_pred], trg_vocab[y]] += 1

    if len(source_labels) > 0:
      fig, ax = plt.subplots(figsize=(30, 10))
      si = np.arange(n_src_vocab+1)
      ti = np.arange(n_trg_vocab+1)
      T, S = np.meshgrid(ti, si)
      plt.pcolormesh(T, S, confusion)
      for i in range(n_src_vocab):
        for j in range(n_trg_vocab):
          plt.text(j, i, confusion[i, j], color='orange')
      ax.set_xticks(ti[1:]-0.5)
      ax.set_yticks(si[1:]-0.5)
      ax.set_xticklabels(sorted(trg_vocab, key=lambda x:trg_vocab[x]))
      ax.set_yticklabels(sorted(src_vocab, key=lambda x:src_vocab[x]))
      plt.xticks(rotation=45)
      plt.colorbar()
      plt.savefig(out_prefix+'_confusion.png')
      plt.close()
      json.dump({'confusion': confusion.tolist(),
                 'src_vocab': sorted(src_vocab, key=lambda x:src_vocab[x]),
                 'trg_vocab': sorted(trg_vocab, key=lambda x:trg_vocab[x])}, 
                open(out_prefix+'_confusion.json', 'w'), indent=2)
    return P_a_norm

  def predict_antecedent(self, P_a):
    span_num = P_a.shape[0] - 1
    antecedents = -1 * np.ones(span_num, dtype=np.int64)
    
    # antecedent prediction
    for idx2 in range(1, span_num+1):
      distances = []
      for idx1 in range(idx2):
        distances.append(kl_divergence(P_a[idx2], P_a[idx1]) + kl_divergence(P_a[idx1], P_a[idx2]))
      antecedents[idx2-1] = np.argmin(distances)-1
    return antecedents                

  def predict_pairwise(self, P_a):
    span_num = P_a.shape[0] - 1
    first, second = zip(*list(itertools.combinations(range(span_num), 2)))
    first = list(first)
    second = list(second)
    labels = []
    for first_idx, second_idx in zip(first, second):
      kl1 = kl_divergence(P_a[first_idx+1], P_a[second_idx+1])
      kl2 = kl_divergence(P_a[second_idx+1], P_a[first_idx+1])
      kl_null1 = kl_divergence(P_a[first_idx+1], P_a[0])
      kl_null2 = kl_divergence(P_a[second_idx+1], P_a[0])
      if kl1 + kl2 < kl_null1 + kl_null2:
        labels.append(1)
      else:
        labels.append(0)
    return labels, first, second

  def retrieve(self, 
               source_features_test, 
               target_features_test, 
               out_file, kbest=10):
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
       
      _, scores[i_utt], _ = self.align_sents(src_feats, trg_feats, alignment_type='image') 

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

def kl_divergence(p, q):
  return np.sum(p * (np.log(np.maximum(p, EPS)) - np.log(np.maximum(q, EPS))))

def to_one_hot(sent, K):
  sent = np.asarray(sent)
  if len(sent.shape) < 2:
    es = np.eye(K)
    sent = np.asarray([es[int(w)] if w < K else 1./K*np.ones(K) for w in sent])
    return sent
  else:
    return sent

def to_pairwise(labels):
  n = labels.shape[0]
  if n <= 1:
    return None
  first, second = zip(*list(itertools.combinations(range(n), 2)))
  first = list(first)
  second = list(second)
  
  pw_labels = (labels[first] == labels[second]) & (labels[first] != 0) & (labels[second] != 0)
  pw_labels = pw_labels.astype(np.int64) 
  return pw_labels

def to_antecedents(labels): 
  n = labels.shape[0]
  antecedents = -1 * np.ones(n, dtype=np.int64)
  for idx in range(n):
    for a_idx in range(idx):
      if labels[idx] == labels[a_idx]:
        antecedents[idx] = a_idx
        break
  return antecedents

def filter_visual_event(text_event_features,
                        visual_event_features,
                        cluster_ids,
                        spans,
                        doc_ids,
                        tokens,
                        text_event_labels,
                        visual_event_labels,
                        ontology_map):
  filtered_event_features = []
  filtered_visual_features = []
  filtered_cluster_ids = []
  filtered_spans = []
  filtered_event_labels = []
  filtered_visual_labels = []
  filtered_tokens = []
  filtered_doc_ids = []
  for inputs in zip(text_event_features,\
                    visual_event_features,\
                    cluster_ids,\
                    spans,\
                    doc_ids,\
                    tokens,\
                    text_event_labels,\
                    visual_event_labels):
    e_feat, v_feat, cluster_id, span, doc_id, token, text_label, visual_label = inputs
    filtered_event_feature = []
    filtered_cluster_id = []
    filtered_span = []
    filtered_event_label = []

    n_spans = len(span)
    is_visual = [0]*n_spans
    for i in range(n_spans):
      for y_t in ontology_map[text_label[i]]:
        if y_t in visual_label:
          filtered_event_feature.append(e_feat[i].tolist())
          filtered_cluster_id.append(cluster_id[i])
          filtered_span.append(span[i])
          filtered_event_label.append(text_label[i])
          break

    if len(filtered_event_feature) > 0:
      filtered_event_feature = np.asarray(filtered_event_feature)
      filtered_cluster_id = np.asarray(filtered_cluster_id)

      filtered_event_features.append(filtered_event_feature)
      filtered_visual_features.append(v_feat)
      filtered_cluster_ids.append(filtered_cluster_id)
      filtered_spans.append(filtered_span)
      filtered_event_labels.append(filtered_event_label)
      filtered_visual_labels.append(visual_label)
      filtered_doc_ids.append(doc_id)
      filtered_tokens.append(token)

  return filtered_event_features,\
         filtered_visual_features,\
         filtered_cluster_ids,\
         filtered_spans,\
         filtered_doc_ids,\
         filtered_tokens,\
         filtered_event_labels,\
         filtered_visual_labels

def load_data(config):
  """
  Returns:
      src_feats_train: a list of arrays of shape (src sent length, src dimension)
      trg_feats_train: a list of arrays of shape (trg sent length, trg dimension)
      src_feats_test: a list of arrays of shape (src sent length, src dimension)
      trg_feats_test: a list of arrays of shape (trg sent length, trg dimension)
  """
  event_mentions = dict()
  text_feat_files = dict()
  doc_to_text_feat = dict()
  documents = dict()
  vocab = dict()
  vocab_freq = dict()
  ontology_map  = json.load(open(os.path.join(config['data_folder'], '../ontology_mapping.json')))
  visual_ontology = json.load(open(os.path.join(config['data_folder'], '../ontology.json')))
  visual_classes = visual_ontology['event']
  visual_feat_files = dict()
  visual_label_files = dict()
  doc_to_visual_feat = dict()
  for split in config['splits']:
    for dataset in config['splits'][split]:
      event_mentions[dataset] = json.load(codecs.open(os.path.join(config['data_folder'], f'{dataset}_events.json'), 'r', 'utf-8'))
      documents[dataset] = json.load(codecs.open(os.path.join(config['data_folder'], f'{dataset}.json')))

      doc_to_text_feat[dataset] = dict()
      doc_to_visual_feat[dataset] = dict()
      for m in event_mentions[dataset]:
        trigger = m['head_lemma']
        if not trigger in vocab:
          vocab[trigger] = len(vocab)
          vocab_freq[trigger] = 1
        else:
          vocab_freq[trigger] += 1

      text_feat_type = config.get('text_feature_type', 'token')
      if text_feat_type == 'token':
        text_feat_files[dataset] = os.path.join(config['data_folder'], f'{dataset}_events_labels.npz')
      elif text_feat_type == 'cluster_probs':
        text_feat_files[dataset] = os.path.join(config['data_folder'], f'{dataset}_events_cluster_probs.npz')
      else:
        raise ValueError

      visual_feat_type = config.get('visual_feature_type', 'frame')
      if visual_feat_type == 'event':
        visual_feat_files[dataset] = os.path.join(config['data_folder'], f'{dataset}_mmaction_event_feat.npz') # XXX 
        visual_label_file = os.path.join(config['data_folder'], f'{dataset}_mmaction_event_feat_labels.npz')
        if os.path.exists(visual_label_file):
          visual_label_files[dataset] = visual_label_file
      elif visual_feat_type == 'frame':
        visual_feat_files[dataset] = os.path.join(config['data_folder'], f'{dataset}_mmaction_feat.npz') 
      else:
        raise ValueError

      text_feats = np.load(text_feat_files[dataset])
      visual_feats = np.load(visual_feat_files[dataset])

      doc_to_text_feat[dataset] = {'_'.join(feat_id.split('/')[-1].split('_')[:-1]):feat_id for feat_id in text_feats}
      doc_to_visual_feat[dataset] = {'_'.join(feat_id.split('/')[-1].split('_')[:-1]):feat_id for feat_id in visual_feats}

  json.dump(vocab_freq, open('vocab_freq.json', 'w'), indent=2)
  vocab_size = len(vocab) 

  label_dict = dict()
  train_num = 0
  test_num = 0
  for split in config['splits']:
    for dataset in config['splits'][split]:
      label_dict[dataset] = dict()
      for m in event_mentions[dataset]:
        if m['doc_id'] in doc_to_visual_feat[dataset]:
          if not m['doc_id'] in label_dict[dataset]:
            label_dict[dataset][m['doc_id']] = dict()
          token = m['head_lemma']
          label_dict[dataset][m['doc_id']][(min(m['tokens_ids']), max(m['tokens_ids']))] = {'token_id': vocab[token],
                                                                                            'cluster_id': m['cluster_id'], 
                                                                                            'type': m['event_type']} 
      if split == 'train':
        train_num += len(label_dict[dataset])
      else:
        test_num += len(label_dict[dataset]) 
  print(f'Vocab size: {vocab_size}')
  print(f'Number of training examples: {train_num}')
  print(f'Number of test examples: {test_num}')

  src_feats_train = []
  trg_feats_train = []
  src_labels_train = []
  trg_labels_train = []
  doc_ids_train = []
  spans_train = []
  cluster_ids_train = []
  tokens_train = [] 
  src_feats_test = []
  trg_feats_test = []
  src_labels_test = []
  trg_labels_test = []
  doc_ids_test = []
  spans_test = []
  cluster_ids_test = []
  tokens_test = []

  for dataset in config['splits']['train']: 
    text_feats = np.load(text_feat_files[dataset])
    visual_feats = np.load(visual_feat_files[dataset])
    visual_labels = None
    cur_trg_feats_train = []
    cur_trg_labels_train = []
    cur_cluster_ids_train = []
    cur_doc_ids_train = []
    cur_spans_train = []
    cur_tokens_train = []
    cur_src_feats_train = []
    cur_src_labels_train = []
    if dataset in visual_label_files: 
      visual_labels = np.load(visual_label_files[dataset])

    for feat_idx, doc_id in enumerate(sorted(label_dict[dataset])): # XXX
      visual_feat_id = doc_to_visual_feat[dataset][doc_id]
      cur_src_feats_train.append(visual_feats[visual_feat_id])
      if visual_labels is not None:
        visual_labels_int = np.argmax(visual_labels[visual_feat_id], axis=-1)
        cur_src_labels_train.append([visual_classes[k] for k in visual_labels_int])      

      spans = sorted(label_dict[dataset][doc_id])
      text_feat_id = doc_to_text_feat[dataset][doc_id]
      trg_sent = text_feats[text_feat_id] 
      trg_labels = [label_dict[dataset][doc_id][span]['type'] for span in spans]
      cluster_ids = [label_dict[dataset][doc_id][span]['cluster_id'] for span in spans]

      cur_trg_feats_train.append(to_one_hot(trg_sent, vocab_size))
      cur_trg_labels_train.append(trg_labels)
      cur_doc_ids_train.append(doc_id)
      cur_spans_train.append(spans)
      cur_cluster_ids_train.append(np.asarray(cluster_ids))
      cur_tokens_train.append([t[2] for t in documents[dataset][doc_id]])
    
    if visual_labels is not None:
      cur_trg_feats_train,\
      cur_src_feats_train,\
      cur_cluster_ids_train,\
      cur_spans_train,\
      cur_doc_ids_train,\
      cur_tokens_train,\
      cur_trg_labels_train,\
      cur_src_labels_train = filter_visual_event(cur_trg_feats_train, 
                                                 cur_src_feats_train,
                                                 cur_cluster_ids_train, 
                                                 cur_spans_train,
                                                 cur_doc_ids_train,
                                                 cur_tokens_train,
                                                 cur_trg_labels_train,
                                                 cur_src_labels_train,
                                                 ontology_map) 
    trg_feats_train.extend(cur_trg_feats_train)
    trg_labels_train.extend(cur_trg_labels_train)
    doc_ids_train.extend(cur_doc_ids_train)
    cluster_ids_train.extend(cur_cluster_ids_train)
    spans_train.extend(cur_spans_train)
    tokens_train.extend(cur_tokens_train)
    src_feats_train.extend(cur_src_feats_train)
    src_labels_train.extend(cur_src_labels_train)

  for dataset in config['splits']['test']:
    text_feats = np.load(text_feat_files[dataset])
    visual_feats = np.load(visual_feat_files[dataset])
    visual_labels = None
    if dataset in visual_label_files: 
      visual_labels = np.load(visual_label_files[dataset])

    for feat_idx, doc_id in enumerate(sorted(label_dict[dataset])): # XXX
      visual_feat_id = doc_to_visual_feat[dataset][doc_id]
      src_feats_test.append(visual_feats[visual_feat_id])
      if visual_labels is not None:
        visual_labels_int = np.argmax(visual_labels[visual_feat_id], axis=-1)
        src_labels_test.append([visual_classes[k] for k in visual_labels_int])

      spans = sorted(label_dict[dataset][doc_id])
      text_feat_id = doc_to_text_feat[dataset][doc_id]
      trg_sent = text_feats[text_feat_id] 
      trg_labels = [label_dict[dataset][doc_id][span]['type'] for span in spans]
      cluster_ids = [label_dict[dataset][doc_id][span]['cluster_id'] for span in spans]
      spans_test.append(spans)
      trg_feats_test.append(to_one_hot(trg_sent, vocab_size))
      trg_labels_test.append(trg_labels)
      doc_ids_test.append(doc_id)
      cluster_ids_test.append(np.asarray(cluster_ids))
      tokens_test.append([t[2] for t in documents[dataset][doc_id]])
  
    if visual_labels is not None:
      trg_feats_test,\
      src_feats_test,\
      cluster_ids_test,\
      spans_test,\
      doc_ids_test,\
      tokens_test,\
      trg_labels_test,\
      src_labels_test = filter_visual_event(trg_feats_test, 
                                            src_feats_test,
                                            cluster_ids_test, 
                                            spans_test,
                                            doc_ids_test,
                                            tokens_test,
                                            trg_labels_test,
                                            src_labels_test,
                                            ontology_map) 
  return src_feats_train, trg_feats_train,\
         src_labels_train, trg_labels_train,\
         src_feats_test, trg_feats_test,\
         src_labels_test, trg_labels_test,\
         doc_ids_train, doc_ids_test,\
         spans_train, spans_test,\
         cluster_ids_train, cluster_ids_test,\
         tokens_train, tokens_test,\
         visual_classes, vocab  


if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', '-c')
  args = parser.parse_args()

  config_file = args.config
  config = pyhocon.ConfigFactory.parse_file(config_file) 
  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  logging.basicConfig(filename=os.path.join(config['model_path'], 'train.log'))
  
  src_feats_train, trg_feats_train,\
  src_labels_train, trg_labels_train,\
  src_feats_test, trg_feats_test,\
  src_labels_test, trg_labels_test,\
  doc_ids_train, doc_ids_test,\
  spans_train, spans_test,\
  cluster_ids_train, cluster_ids_test,\
  tokens_train, tokens_test,\
  src_vocab, trg_vocab = load_data(config)
  
  ## Model training
  config['vocab'] = sorted(trg_vocab, key=lambda x:trg_vocab[x])
  aligner = FullyContinuousMixtureAligner(src_feats_train+src_feats_test, 
                                          trg_feats_train+trg_feats_test,
                                          configs=config)
  aligner.trainEM(5, os.path.join(config['model_path'], 'mixture')) # XXX 
 
  '''
  example_idx = 0
  for doc_id in doc_ids_test:
    if doc_id == '7rI02PKM-Do':
      break
    example_idx += 1
  print(doc_id, tokens_test[example_idx], src_labels_test[example_idx]) # XXX
  '''
  P_a_norm = aligner.align(src_feats_test,
                           trg_feats_test,
                           src_labels_test,
                           trg_labels_test,
                           out_prefix=os.path.join(config['model_path'], 
                                                   'alignment'),
                           alignment_type='text')  
  _ = aligner.align(src_feats_test,
                    trg_feats_test,
                    src_labels_test,
                    trg_labels_test,
                    out_prefix=os.path.join(config['model_path'], 
                                            'alignment'),
                    alignment_type='image')  

  antecedents = [aligner.predict_antecedent(P_a) for P_a in P_a_norm]
  pred_labels, first, second = zip(*[aligner.predict_pairwise(P_a) for P_a in P_a_norm if P_a.shape[0] > 2])
  first = list(first)
  second = list(second)

  # Save predictions
  predictions = []
  for doc_id, t, s, a, y, first_idxs, second_idxs in zip(doc_ids_test, 
                                                         tokens_test, 
                                                         spans_test, 
                                                         antecedents, 
                                                         pred_labels, 
                                                         first, 
                                                         second):
    predictions.append({'doc_id': doc_id,
                        'tokens': t,
                        'spans': s,
                        'antecedents': a.tolist(),
                        'first': first_idxs,
                        'second': second_idxs,
                        'labels': y})
  json.dump(predictions, open(os.path.join(config['model_path'], 'predictions.json'), 'w'), indent=2)

  ## Test and evaluation
  conll_eval = CoNLLEvaluation()
  pred_labels = [torch.LongTensor(y) for y in pred_labels]
  gold_labels = [torch.LongTensor(to_pairwise(c)) for c in cluster_ids_test if c.shape[0] > 1]
  pred_labels = torch.cat(pred_labels)
  gold_labels = torch.cat(gold_labels) 
  assert len(pred_labels) == len(gold_labels)

  # Compute pairwise scores
  pred_labels_baseline = torch.ones(pred_labels.size(), dtype=torch.int)
  pairwise_eval_baseline = Evaluation(pred_labels_baseline, gold_labels) 
  print(f'BASELINE (Predict Dominant Class), Pairwise - Precision: {pairwise_eval_baseline.get_precision()}, Recall: {pairwise_eval_baseline.get_recall()}, F1: {pairwise_eval_baseline.get_f1()}')

  pairwise_eval = Evaluation(pred_labels, gold_labels)
  print(f'Pairwise - Precision: {pairwise_eval.get_precision()}, Recall: {pairwise_eval.get_recall()}, F1: {pairwise_eval.get_f1()}')
  logger.info(f'Pairwise precision: {pairwise_eval.get_precision()}, recall: {pairwise_eval.get_recall()}, F1: {pairwise_eval.get_f1()}')
  

  # Compute CoNLL scores and save readable predictions
  f_out = open(os.path.join(config['model_path'], 'prediction.readable'), 'w')
  for doc_id, token, span, antecedent, cluster_id in zip(doc_ids_test, tokens_test, spans_test, antecedents, cluster_ids_test):
    pred_clusters, gold_clusters = conll_eval(torch.LongTensor(span),
                                              torch.LongTensor(antecedent),
                                              torch.LongTensor(span),
                                              torch.LongTensor(cluster_id)) 
    pred_clusters_str, gold_clusters_str = conll_eval.make_output_readable(pred_clusters, gold_clusters, token) 
    token_str = ' '.join(token)
    f_out.write(f'{doc_id}: {token_str}\n')
    f_out.write(f'Pred: {pred_clusters_str}\n')
    f_out.write(f'Gold: {gold_clusters_str}\n\n')
  f_out.close() 
  
  muc, b_cubed, ceafe, avg = conll_eval.get_metrics()
  conll_metrics = muc+b_cubed+ceafe+avg
  print('MUC - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
        'Bcubed - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
        'CEAFe - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
        'CoNLL - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(*conll_metrics)) 
  logger.info('MUC - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'Bcubed - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CEAFe - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CoNLL - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(*conll_metrics))

  aligner.retrieve(src_feats_test, trg_feats_test, os.path.join(config['model_path'], 'retrieval'))

