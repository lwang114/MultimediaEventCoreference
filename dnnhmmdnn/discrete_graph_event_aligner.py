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
import itertools
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from scipy.special import logsumexp
from region_vgmm import *
from negative_square import NegativeSquare
from evaluator import Evaluation, CoNLLEvaluation
np.random.seed(2)

logger = logging.getLogger(__name__)
EPS = 1e-15
class GraphMixtureEventAligner(object):
  """An alignment model based on Brown et. al., 1993. capable of 
  modeling visual-linguistic event-argument structures
  """
  def __init__(self, 
               action_features_train,
               object_features_train, 
               event_features_train, 
               entity_features_train,
               action_to_object_mappings,
               event_to_argument_mappings, 
               configs):
    self.Ke = configs.get('n_event_vocab', 500)
    self.Ka = configs.get('n_entity_vocab', 500)
    self.Kv = action_features_train[0].shape[-1] # configs.get('n_action_vocab', 33)
    self.Ko = object_features_train[0].shape[-1] # configs.get('n_object_vocab', 7)    
    self.use_null = configs.get('use_null', False)

    self.pretrained_action_model = configs.get('pretrained_action_model', None)
    self.pretrained_object_model = configs.get('pretrained_object_model', None)
    self.pretrained_translateprob = configs.get('pretrained_event_translateprob', None)
    self.pretrained_translateprob = configs.get('pretrained_entity_translateprob', None)

    var = configs.get('var', 160.) # XXX   
    print(f'n_event_vocab={self.Ke}, n_entity_vocab={self.Ka}') 
    print(f'n_action_vocab={self.Kv}, n_object_vocab={self.Ko}')
    logger.info(f'n_event_vocab={self.Ke}, n_entity_vocab={self.Ka}')
    logger.info(f'n_action_vocab={self.Kv}, n_object_vocab={self.Ko}')

    self.alpha = configs.get('alpha', 0.)
    if event_features_train[0].ndim <= 1:
      self.event_embedding_dim = 1 
    else:
      self.event_embedding_dim = event_features_train[0].shape[-1]

    if entity_features_train[0].ndim <= 1:
      self.entity_embedding_dim = 1
    else:
      self.entity_embedding_dim = entity_features_train[0].shape[-1] 
    print(f'event embedding dimension={self.event_embedding_dim}, '
          f'entity embedding dimension={self.entity_embedding_dim}')

    self.action_vec_ids_train = []
    self.object_vec_ids_train = []
    start_index_action = 0
    start_index_object = 0
    for ex, (act_feat, obj_feat) in enumerate(zip(action_features_train, object_features_train)):
      if self.use_null:
        if self.event_embedding_dim == 1:
          event_features_train[ex] = event_features_train[ex] + [self.Ke-1]
      
        if self.entity_embedding_dim == 1:
          entity_features_train[ex] = entity_features_train[ex] + [self.Kn-1]

      action_vec_ids = []
      for t in range(len(act_feat)):
        action_vec_ids.append(start_index_action+t)

      object_vec_ids = []
      for t in range(len(obj_feat)):
        object_vec_ids.append(start_index_object+t)

      start_index_action += len(act_feat)
      start_index_object += len(obj_feat)
      
      self.action_vec_ids_train.append(action_vec_ids)
      self.object_vec_ids_train.append(object_vec_ids)
    
    self.action_model = RegionVGMM(np.concatenate(action_features_train, axis=0),
                                   2, # self.Kv,
                                   var=var,
                                   vec_ids=self.action_vec_ids_train,
                                   pretrained_model=self.pretrained_action_model)
    
    self.object_model = RegionVGMM(np.concatenate(object_features_train, axis=0),
                                   2, # self.Ko,
                                   var=var,
                                   vec_ids=self.object_vec_ids_train,
                                   pretrained_model=self.pretrained_object_model)

    self.action_feats = self.action_model.X
    self.object_feats = self.object_model.X
    self.vo_maps = action_to_object_mappings

    self.event_feats = event_features_train
    self.entity_feats = entity_features_train
    self.ea_maps = event_to_argument_mappings

    if self.pretrained_translateprob:
      self.P_ev = np.load(self.pretrained_event_translateprob)
      self.P_ao = np.load(self.pretrained_entity_translateprob)
      print('Loaded pretrained translation probabilities')
    else:
      self.P_ev = 1./self.Ka * np.ones((self.Ke, self.Kv))
      self.P_ao = 1./self.Ko * np.ones((self.Ka, self.Ko))
    self.ev_counts = np.zeros((self.Ke, self.Kv))
    self.ao_counts = np.zeros((self.Ka, self.Ko))

  def compute_forward_event_probs(self, 
                                  event_probs, action_probs,
                                  argument_probs, object_probs):
    Ne = event_probs.shape[0]
    Nv = action_probs.shape[0]
    A = np.ones((Ne, Ne)) / max(Ne, 1)
    init = np.ones(Ne) / max(Ne, 1)
    forward_probs = np.zeros((Nv, Ne, self.Ke))
    scales = np.zeros((Nv,))

    # (Nv, Ke)
    probs_v_t_given_e = action_probs @ self.P_ev.T 
    
    # Compute P(O_t|A_i)
    # (Nv, Ne)
    probs_o_t_given_a_i = [[self.compute_forward_probs(
                                object_probs[t], 
                                argument_probs[i], 
                                self.P_ao, self.Ka
                                )[1].prod() for i in range(Ne)] 
                                                for t in range(Nv)]
    probs_o_t_given_a_i = np.asarray(probs_o_t_given_a_i)

    # (Nv, Ne, Ke)
    probs_vo_t_given_ea_i = probs_v_t_given_e[:, np.newaxis] * probs_o_t_given_a_i[:, :, np.newaxis]

    forward_probs[0] = np.tile(init[:, np.newaxis], (1, self.Ke)) *\
                       event_probs * probs_vo_t_given_ea_i[0]
    scales[0] = np.sum(forward_probs[0])
    forward_probs[0] /= np.maximum(scales[0], EPS)
    A_diag = np.diag(np.diag(A))
    A_offdiag = A - A_diag
    for t in range(Nv-1):
      # (Ne, Ke)
      probs_vo_e_t_given_ea_i = event_probs * probs_vo_t_given_ea_i[t+1]
      forward_probs[t+1] += (A_diag @ forward_probs[t]) * probs_vo_t_given_ea_i[t+1]
      forward_probs[t+1] += ((A_offdiag.T @ np.sum(forward_probs[t], axis=-1)) * probs_vo_e_t_given_ea_i.T).T    
      scales[t+1] = np.sum(forward_probs[t+1])
      forward_probs[t+1] /= max(scales[t+1], EPS)
    return forward_probs, scales

  def compute_backward_event_probs(self,
                                   event_probs, action_probs,
                                   argument_probs, object_probs, scales):
    Ne = event_probs.shape[0]
    Nv = action_probs.shape[0]
    A = np.ones((Ne, Ne)) / max(Ne, 1)
    init = np.ones(Ne) / max(Ne, 1)
    backward_probs = np.zeros((Nv, Ne, self.Ke))
    backward_probs[Nv-1] = 1. / max(scales[Nv-1], EPS)
    
    A_diag = np.diag(np.diag(A))
    A_offdiag = A - A_diag

    # (Nv, Ke)
    probs_v_t_given_e = action_probs @ self.P_ev.T 
    
    # Compute P(O_t|A_i)
    # (Nv, Ne)
    probs_o_t_given_a_i = [[self.compute_forward_probs(
                                object_probs[t], 
                                argument_probs[i], 
                                self.P_ao, self.Ka
                                )[1].prod() for i in range(Ne)] 
                                                for t in range(Nv)]
    probs_o_t_given_a_i = np.asarray(probs_o_t_given_a_i)
    # (Nv, Ne, Ke)
    probs_vo_t_given_ea_i = probs_v_t_given_e[:, np.newaxis] * probs_o_t_given_a_i[:, :, np.newaxis]

    for t in range(Nv-1, 0, -1):
      # (Ne, Ke)
      probs_vo_e_t_given_ea_i = event_probs * probs_vo_t_given_ea_i[t]
      backward_probs[t-1] = A_diag @ (backward_probs[t] * probs_vo_t_given_ea_i[t])
      backward_probs[t-1] += np.tile(A_offdiag @ np.sum(backward_probs[t] * probs_vo_e_t_given_ea_i, axis=-1)[:, np.newaxis], (1, self.Ke))
      backward_probs[t-1] /= max(scales[t-1], EPS)
    return backward_probs

  def compute_forward_probs(self, src_sent, trg_sent, P_ts, Kt):
    L = trg_sent.shape[0]
    T = src_sent.shape[0]
    if L == 0 and T > 0:
      return None, np.ones(T,)
    if L == 0 or T == 0:
      return None, np.ones(1,)

    A = np.ones((L, L)) / max(L, 1)
    init = np.ones(L) / max(L, 1)
    forward_probs = np.zeros((T, L, Kt))
    scales = np.zeros((T,))
    
    probs_x_t_given_z = src_sent @ P_ts.T
    forward_probs[0] = np.tile(init[:, np.newaxis], (1, Kt)) * trg_sent * probs_x_t_given_z[0] 
    scales[0] = np.sum(forward_probs[0])
    forward_probs[0] /= np.maximum(scales[0], EPS)
    A_diag = np.diag(np.diag(A))
    A_offdiag = A - A_diag
    for t in range(T-1):
      probs_x_t_z_given_y = trg_sent * probs_x_t_given_z[t+1]
      forward_probs[t+1] += (A_diag @ forward_probs[t]) * probs_x_t_given_z[t+1]
      forward_probs[t+1] += ((A_offdiag.T @ np.sum(forward_probs[t], axis=-1)) * probs_x_t_z_given_y.T).T
      scales[t+1] = np.sum(forward_probs[t+1])
      forward_probs[t+1] /= max(scales[t+1], EPS)
    return forward_probs, scales
      
  def compute_backward_probs(self, src_sent, trg_sent, P_ts, Kt, scales):
    T = src_sent.shape[0]
    L = trg_sent.shape[0]
    if L == 0 or T == 0:
      return None

    A = np.ones((L, L)) / max(L, 1)
    init = np.ones(L) / max(L, 1)
    backward_probs = np.zeros((T, L, Kt))
    backward_probs[T-1] = 1. / max(scales[T-1], EPS)

    A_diag = np.diag(np.diag(A))
    A_offdiag = A - A_diag
    probs_x_t_given_z = src_sent @ P_ts.T
    for t in range(T-1, 0, -1):
      probs_x_t_z_given_y = trg_sent * probs_x_t_given_z[t]
      backward_probs[t-1] = A_diag @ (backward_probs[t] * probs_x_t_given_z[t])
      backward_probs[t-1] += np.tile(A_offdiag @ np.sum(backward_probs[t] * probs_x_t_z_given_y, axis=-1)[:, np.newaxis], (1, Kt))
      backward_probs[t-1] /= max(scales[t-1], EPS)
    return backward_probs
    
  def update_counts(self):
    # Update alignment counts
    log_probs = []
    self.ev_counts[:] = 0.
    self.ao_counts[:] = 0.
    for i in range(len(self.event_feats)):
      C_ev, C_ao, log_prob_i = self.update_counts_i(i)
      self.ev_counts += C_ev
      self.ao_counts += C_ao
      log_probs.append(log_prob_i)

    P_ev, P_ao = self.translate_probs()
    self.P_ev = deepcopy(P_ev) 
    self.P_ao = deepcopy(P_ao)
    return np.mean(log_probs)

  def update_counts_i(self, i):
    P_e = self.event_feats[i] 
    P_v = self.action_feats[self.action_vec_ids_train[i]] # np.exp(self.action_model.log_prob_z(i, normalize=False))
    P_a = self.entity_feats[i]
    P_o = self.object_feats[self.object_vec_ids_train[i]] # np.exp(self.object_model.log_prob_z(i, normalize=False))
              
    P_e = to_one_hot(P_e, self.Ke)
    P_v = to_one_hot(P_v, self.Kv)
    P_a = reshape_by_event(to_one_hot(P_a, self.Ka), self.ea_maps[i]) 
    P_o = reshape_by_event(to_one_hot(P_o, self.Ko), self.vo_maps[i])
   
    Nv = P_v.shape[0]
    Ne = P_e.shape[0]

    # Compute event-to-action counts
    # (Nv, Ne, Ke)
    F_ev, scales_ev = self.compute_forward_event_probs(P_e, P_v, P_a, P_o)
    # (Nv, Ne, Ke)
    B_ev = self.compute_backward_event_probs(P_e, P_v, P_a, P_o, scales_ev)

    norm_factor = np.sum(F_ev * B_ev, axis=(1, 2), keepdims=True) 
    # (Nv, Ne, Ke)
    new_ev_counts = F_ev * B_ev / np.maximum(norm_factor, EPS) 
    # (Ke, Kv)
    C_ev = np.sum(new_ev_counts, axis=1).T @ (P_v / np.maximum(np.sum(P_v, axis=1, keepdims=True), EPS))

    # Compute argument-to-object counts 
    C_ao = np.zeros((self.Ka, self.Ko))    
    for i_v in range(Nv):
      for i_e in range(Ne):
        if P_o[i_v].shape[0] == 0 or P_a[i_e].shape[0] == 0:
          continue
        F_ao, scales_ao = self.compute_forward_probs(P_o[i_v], P_a[i_e], self.P_ao, self.Ka)
        B_ao = self.compute_backward_probs(P_o[i_v], P_a[i_e], self.P_ao, self.Ka, scales_ao)
        norm_factor = np.sum(F_ao * B_ao, axis=(1, 2), keepdims=True)
        # (No, Na, Ka)
        new_ao_counts = new_ev_counts[i_v, i_e].sum() * F_ao * B_ao / np.maximum(norm_factor, EPS)
        # (Ka, Ko)
        C_ao += np.sum(new_ao_counts, axis=1).T @ (P_o[i_v] / np.maximum(np.sum(P_o[i_v], axis=1, keepdims=True), EPS))
    
    log_prob = np.log(np.maximum(scales_ev, EPS)).sum()
    return C_ev, C_ao, log_prob

  def update_components(self):
    means_v_new = np.zeros(self.action_model.means.shape)
    means_o_new = np.zeros(self.object_model.means.shape)

    counts_v = np.zeros(self.Kv,)
    counts_o = np.zeros(self.Ko,)
    for i, (P_e, P_a, v_feat, o_feat) in enumerate(zip(self.event_feats, 
                                                       self.entity_feats,
                                                       self.action_feats,
                                                       self.object_feats)):
      if len(e_feat) == 0 or len(v_feat) == 0:
        continue 
      
      # (Kv,)
      prob_z_given_e_all = self.prob_v_given_e_all(P_e)
      # (Nv, Kv)
      prob_z_given_v = np.exp(self.action_model.log_prob_z(i))
      # (Ko,)
      prob_z_given_a_all = self.prob_o_given_a_all(P_a)
      # (No, Ko)
      prob_z_given_o = np.exp(self.object_model.log_prob_z(i))

      # (Nv, Kv)
      post_zv = prob_z_given_e_all * prob_z_given_v
      post_zv /= np.maximum(np.sum(post_zv, axis=1, keepdims=True), EPS)
      # (No, Ko)
      post_zo = prob_z_given_a_all * prob_z_given_o
      post_zo /= np.maximum(np.sum(post_zo, axis=1, keepdims=True), EPS) 
      
      # Update target word counts of the target model
      v_indices = self.action_vec_ids_train[i]
      o_indices = self.object_vec_ids_train[i]

      means_v_new += np.sum(post_zv[:, :, np.newaxis] * self.action_model.X[v_indices, np.newaxis], axis=0)
      means_o_new += np.sum(post_zo[:, :, np.newaxis] * self.object_model.X[o_indices, np.newaxis], axis=0)
      counts_v += np.sum(post_zv, axis=0)
      counts_o += np.sum(post_zo, axis=0)

    self.action_model.means = deepcopy(means_v_new / np.maximum(counts_v[:, np.newaxis], EPS)) 
    self.object_model.means = deepcopy(means_o_new / np.maximum(counts_o[:, np.newaxis], EPS))

  def trainEM(self, n_iter, out_file):
    for i_iter in range(n_iter):
      log_prob = self.update_counts()
      # self.update_components()
      print('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      logger.info('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      if (i_iter + 1) % 5 == 0:
        with open('{}_action_means.json'.format(out_file), 'w') as fv,\
             open('{}_object_means.json'.format(out_file), 'w') as fo,\
             open('{}_ev_transprob.json'.format(out_file), 'w') as fe,\
             open('{}_ao_transprob.json'.format(out_file), 'w') as fa:
          json.dump(self.action_model.means.tolist(), fv, indent=4, sort_keys=True)
          json.dump(self.object_model.means.tolist(), fo, indent=4, sort_keys=True)
          json.dump(self.P_ev.tolist(), fe, indent=4, sort_keys=True)
          json.dump(self.P_ao.tolist(), fa, indent=4, sort_keys=True)         

  def translate_probs(self):
    P_ev = (self.alpha / self.Kv + self.ev_counts) /\
            np.maximum(self.alpha + np.sum(self.ev_counts, axis=-1, keepdims=True), EPS)
    P_ao = (self.alpha / self.Ko + self.ao_counts) /\
            np.maximum(self.alpha + np.sum(self.ao_counts, axis=-1, keepdims=True), EPS)
    return P_ev, P_ao 
  
  def prob_v_given_e_all(self, P_e):
    P_e = to_one_hot(P_e, self.Ke)
    return np.mean(P_e @ self.P_ev, axis=0)
  
  def prob_o_given_a_all(self, P_a):
    P_a = to_one_hot(P_a, self.Ka)
    return np.mean(P_a @ self.P_ao, axis=0) 
    
  def align_sents(self, 
                  action_feats_test, 
                  object_feats_test,
                  event_feats_test, 
                  entity_feats_test,
                  vo_maps_test,
                  ea_maps_test,
                  alignment_type='text'): 
    alignments = []
    scores = []
    for v_feat, o_feat, P_e, P_a, vo_map, ea_map in zip(action_feats_test, object_feats_test,
                                                        event_feats_test, entity_feats_test,
                                                        vo_maps_test, ea_maps_test):
      P_v = v_feat 
      # [np.exp(self.action_model.log_prob_z_given_X(v_feat[i], normalize=False))\
      #     for i in range(len(v_feat))]
      P_o = o_feat 
      # np.exp(self.object_model.log_prob_z_given_X(o_feat[i], normalize=False))\
      #   for i in range(len(o_feat))]
      P_e = to_one_hot(P_e, self.Ke)
      P_a = reshape_by_event(to_one_hot(P_a, self.Ka), ea_map)
      P_v = to_one_hot(P_v, self.Kv)
      P_o = reshape_by_event(to_one_hot(P_o, self.Ko), vo_map)
      P_v_null = P_v.mean() * np.ones((P_e.shape[0], 1))
      P_o_null = np.prod([P_o_i.mean() for P_o_i in P_o]) * np.ones((P_e.shape[0], 1))
      
      Nv = P_v.shape[0]
      Ne = P_e.shape[0]

      # (Ne, Nv)
      P_align_ev = P_e @ self.P_ev @ P_v.T
      # (Ne, Nv)
      P_align_ao = np.asarray([[self.compute_forward_probs(P_o[t], P_a[i], 
                                                self.P_ao, self.Ka)[1].prod()\
                                                for t in range(Nv)] 
                                                    for i in range(Ne)]) 
      P_align_ao = np.asarray(P_align_ao)
      P_align = np.concatenate([P_v_null * P_o_null, P_align_ev * P_align_ao], axis=1)


      scores.append(np.prod(np.max(P_align, axis=0)))
      if alignment_type == 'text':
        alignments.append(np.argmax(P_align, axis=1))       
      elif alignment_type == 'image':
        alignments.append(np.argmax(P_align, axis=0)) 
      else:
        raise ValueError('Alignment type not implemented')

    return alignments, np.asarray(scores)

  def align(self,
            action_feats,
            object_feats,
            event_feats,
            entity_feats,
            action_labels,
            object_labels,
            event_labels,
            entity_labels,
            vo_maps,
            ea_maps,
            alignment_type='image',
            out_prefix='align'):
    action_vocab = dict()
    object_vocab = dict()
    event_vocab = dict()
    entity_vocab = dict()
    for action_label in action_labels:
      for y in action_label:
        if not y in action_vocab:
          action_vocab[y] = len(action_vocab)

    for object_label in object_labels:
      for ys in object_label:
        for y in ys:
          if not y in object_vocab:
            object_vocab[y] = len(object_vocab)

    for event_label in event_labels:
      for y in event_label:
        if not y in event_vocab:
          event_vocab[y] = len(event_vocab)
  
    for entity_label in entity_labels:
      for ys in entity_label:
        for y in ys:
          if not y in entity_vocab:
            entity_vocab[y] = len(entity_vocab)

    n_action_vocab = len(action_vocab)
    n_event_vocab = len(event_vocab)
    alignments, _ = self.align_sents(action_feats,
                                     object_feats,
                                     event_feats,
                                     entity_feats,
                                     vo_maps,
                                     ea_maps,
                                     alignment_type=alignment_type)
    confusion = np.zeros((n_action_vocab, n_event_vocab))
    for action_label, event_label, alignment in zip(action_labels,
                                                    event_labels,
                                                    alignments):
      if alignment_type == 'image':
        for action_idx, y in enumerate(action_label):
          y_pred = event_label[alignment[action_idx]]
          confusion[action_vocab[y], event_vocab[y_pred]] += 1

    fig, ax = plt.subplots(figsize=(30, 10))
    si = np.arange(n_action_vocab+1)
    ti = np.arange(n_event_vocab+1)
    T, S = np.meshgrid(ti, si)
    plt.pcolormesh(T, S, confusion)
    for i in range(n_action_vocab):
      for j in range(n_event_vocab):
        plt.text(j, i, confusion[i, j], color='orange')
    ax.set_xticks(ti[1:]-0.5)
    ax.set_yticks(si[1:]-0.5)
    ax.set_xticklabels(sorted(event_vocab, key=lambda x:event_vocab[x]))
    ax.set_yticklabels(sorted(action_vocab, key=lambda x:action_vocab[x]))
    plt.xticks(rotation=45)
    plt.colorbar()
    plt.savefig(out_prefix+'_confusion.png')
    plt.close()
    json.dump({'confusion': confusion.tolist(),
               'action_vocab': sorted(action_vocab, key=lambda x:action_vocab[x]),
               'event_vocab': sorted(event_vocab, key=lambda x:event_vocab[x])}, 
              open(out_prefix+'_confusion.json', 'w'), indent=2)


  def retrieve(self, 
               action_features_test, 
               object_features_test,
               event_features_test, 
               entity_features_test,
               vo_maps_test,
               ea_maps_test,
               out_file, kbest=10):
    n = len(action_features_test)
    print(n)
    scores = np.zeros((n, n))
    for i_utt in range(n):
      if self.use_null:
        action_feats = [action_features_test[i_utt] for _ in range(n)] 
        object_feats = [object_features_test[i_utt] for _ in range(n)]
        event_feats = [[self.Ke - 1] + event_features_test[j_utt] for j_utt in range(n)]
        entity_feats = [[self.Ka - 1] + entity_features_test[j_utt] for j_utt in range(n)]
        # TODO
        raise NotImplementedError
      else:
        action_feats = [action_features_test[i_utt] for _ in range(n)]
        object_feats = [object_features_test[i_utt] for _ in range(n)]
        event_feats = [event_features_test[j_utt] for j_utt in range(n)] 
        entity_feats = [entity_features_test[j_utt] for j_utt in range(n)]
        vo_map = [vo_maps_test[i_utt] for _ in range(n)]
        ea_map = [ea_maps_test[j_utt] for j_utt in range(n)]
       
      _, scores[i_utt] = self.align_sents(action_feats,
                                          object_feats,
                                          event_feats, 
                                          entity_feats, 
                                          vo_map,
                                          ea_map,
                                          alignment_type='image') 

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


def reshape_by_event(x, mappings):
  return [m @ x for m in mappings] 

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

def load_text_features(config, vocab, vocab_entity, doc_to_feat, split):
  lemmatizer = WordNetLemmatizer() 
  event_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_events.json'), 'r', 'utf-8'))
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}.json')))
  vocab_size = len(vocab)
  vocab_entity_size = len(vocab_entity)

  label_dicts = {}
  event_feats = []
  entity_feats = []
  event_types = []
  entity_types = []
  ea_maps_all = []
  doc_ids = []
  spans_all = []
  spans_entity_all = []
  cluster_ids_all = []
  tokens_all = [] 

  for m in event_mentions:
    if m['doc_id'] in doc_to_feat:
      if not m['doc_id'] in label_dicts:
        label_dicts[m['doc_id']] = {}
      token = lemmatizer.lemmatize(m['tokens'].lower(), pos='v')
      span = (min(m['tokens_ids']), max(m['tokens_ids']))
      label_dicts[m['doc_id']][span] = {'token_id': vocab[token],
                                        'cluster_id': m['cluster_id'],
                                        'type': m['event_type'],
                                        'arguments': {}} 
      
      for a in m['arguments']:
        if 'text' in a:
          a_token = a['text']
          a_span = (a['start'], a['end'])
        else:
          a_token = a['tokens']
          a_span = (min(a['tokens_ids']), max(a['tokens_ids']))
        a_token = lemmatizer.lemmatize(a_token.lower())
        label_dicts[m['doc_id']][span]['arguments'][a_span] = {'token_id': vocab_entity[a_token],
                                                               'type': a['role']}

  for feat_idx, doc_id in enumerate(sorted(label_dicts)): # XXX
    feat_id = doc_to_feat[doc_id]
    label_dict = label_dicts[doc_id]
    spans = sorted(label_dict)
    a_spans = [a_span for span in spans for a_span in sorted(label_dict[span]['arguments'])]
    events = [label_dict[span]['token_id'] for span in spans]
    entities = [label_dict[span]['arguments'][a_span]['token_id'] for span in spans for a_span in sorted(label_dict[span]['arguments'])]
    event_types.append([label_dict[span]['type'] for span in spans])
    entity_types.append([label_dict[span]['arguments'][a_span]['type'] for span in spans for a_span in sorted(label_dict[span]['arguments'])])

    cluster_ids = [label_dict[span]['cluster_id'] for span in spans]
    
    ea_maps = []
    entity_idx = 0
    for span in spans:
      a_spans = sorted(label_dict[span]['arguments'])
      ea_map = np.zeros((len(a_spans), len(entities)))
      for a_idx, a_span in enumerate(a_spans):
        ea_map[a_idx, entity_idx] = 1.
        entity_idx += 1
      ea_maps.append(ea_map)

    event_feats.append(to_one_hot(events, vocab_size))
    entity_feats.append(to_one_hot(entities, vocab_entity_size))
    ea_maps_all.append(ea_maps)
    doc_ids.append(doc_id)
    spans_all.append(spans)
    spans_entity_all.append(a_spans)    
    cluster_ids_all.append(np.asarray(cluster_ids))
    tokens_all.append([t[2] for t in doc_train[doc_id]])
  return event_feats,\
         entity_feats,\
         event_types,\
         entity_types,\
         ea_maps_all,\
         doc_ids,\
         spans_all,\
         spans_entity_all,\
         cluster_ids_all,\
         tokens_all,\
         label_dicts


def load_visual_features(config, label_dicts, action_classes, object_classes, split):
  action_feats_npz = np.load(os.path.join(config['data_folder'], config[f'action_feature_{split}']))
  object_feats_npz = np.load(os.path.join(config['data_folder'], config[f'object_feature_{split}']))
  action_labels_npz = np.load(os.path.join(config['data_folder'], config[f'action_label_{split}']))
  object_labels_npz = np.load(os.path.join(config['data_folder'], config[f'object_label_{split}']))

  doc_to_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in action_feats_npz}

  action_feats = []
  object_feats = []
  action_types = []
  object_types = []
  vo_maps = []
  for feat_idx, doc_id in enumerate(sorted(label_dicts)): # XXX
    feat_id = doc_to_feat[doc_id]
    label_dict = label_dicts[doc_id]
    action_feats.append(action_feats_npz[feat_id])
    action_type_int = np.argmax(action_labels_npz[feat_id], axis=-1)
    action_types.append([action_classes[k] for k in action_type_int])

    o_feats = []
    o_types = []
    cur_vo_maps = []
    for o_feat in object_feats_npz[feat_id]:
      n_roles = (o_feat.mean(-1) != -1).sum()
      o_feats.append(to_one_hot(o_feat[:n_roles, 1], len(object_classes))) # XXX
    o_feats = np.concatenate(o_feats, axis=0)

    o_idx = 0
    for o_feat, o_label in zip(object_feats_npz[feat_id], object_labels_npz[feat_id]):
      n_roles = (o_feat.mean(-1) != -1).sum()
      vo_map = np.zeros((n_roles, o_feats.shape[0]))
      o_label_int = o_feat[:n_roles, 1].astype(int)
      o_types.extend([object_classes[k] for k in o_label_int])
      for r_idx in range(n_roles):
        vo_map[r_idx, o_idx] = 1.
        o_idx += 1
      cur_vo_maps.append(vo_map)
      
    object_feats.append(o_feats)
    object_types.append(o_types)
    vo_maps.append(cur_vo_maps)
  return action_feats, object_feats, action_types, object_types, vo_maps


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
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], 'train.json')))
  event_mentions_test = json.load(codecs.open(os.path.join(config['data_folder'], 'test_events.json'), 'r', 'utf-8'))
  doc_test = json.load(codecs.open(os.path.join(config['data_folder'], 'test.json')))

  action_feats_train_npz = np.load(os.path.join(config['data_folder'], config['action_feature_train']))
  action_feats_test_npz = np.load(os.path.join(config['data_folder'], config['action_feature_test']))
  visual_class_dict = json.load(open(os.path.join(config['data_folder'], '../ontology.json')))
  action_classes = visual_class_dict['event']
  object_classes = visual_class_dict['arguments'] # XXX

  doc_to_feat_train = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in action_feats_train_npz}
  doc_to_feat_test = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in action_feats_test_npz}

  vocab = dict()
  vocab_freq = dict()
  vocab_entity = dict()
  vocab_entity_freq = dict()
  for m in event_mentions_train + event_mentions_test:
    trigger = m['tokens']
    trigger = lemmatizer.lemmatize(trigger.lower(), pos='v')
    if not trigger in vocab:
      vocab[trigger] = len(vocab)
      vocab_freq[trigger] = 1
    else:
      vocab_freq[trigger] += 1
    
    for a in m['arguments']:
      if 'text' in a:
        argument = a['text']
      else:
        argument = a['tokens']
      argument = lemmatizer.lemmatize(argument.lower())
      if not argument in vocab_entity:
        vocab_entity[argument] = len(vocab_entity)
        vocab_entity_freq[argument] = len(vocab_entity)
      else:
        vocab_entity_freq[argument] += 1

  json.dump(vocab_freq, open('vocab_freq.json', 'w'), indent=2)
  json.dump(vocab_entity_freq, open('vocab_entity_freq.json', 'w'), indent=2)

  vocab_size = len(vocab_freq)
  vocab_entity_size = len(vocab_entity_freq)
  print(f'Vocab size: {vocab_size}, vocab entity size: {vocab_entity_size}')

  event_feats_train,\
  entity_feats_train,\
  _, _,\
  ea_maps_train,\
  doc_ids_train,\
  spans_train,\
  spans_entity_train,\
  cluster_ids_train,\
  tokens_train,\
  label_dict_train = load_text_features(config, vocab, vocab_entity, doc_to_feat_train, split='train')
  print(f'Number of training examples: {len(label_dict_train)}')

  event_feats_test,\
  entity_feats_test,\
  event_types_test,\
  entity_types_test,\
  ea_maps_test,\
  doc_ids_test,\
  spans_test,\
  spans_entity_test,\
  cluster_ids_test,\
  tokens_test,\
  label_dict_test = load_text_features(config, vocab, vocab_entity, doc_to_feat_test, split='test')
  print(f'Number of test examples: {len(label_dict_test)}')

  action_feats_train,\
  object_feats_train,\
  _, _,\
  vo_maps_train = load_visual_features(config, label_dict_train, action_classes, object_classes, split='train')
  action_feats_test,\
  object_feats_test,\
  action_types_test,\
  object_types_test,\
  vo_maps_test = load_visual_features(config, label_dict_test, action_classes, object_classes, split='test')

  return event_feats_train,\
         entity_feats_train,\
         ea_maps_train,\
         doc_ids_train,\
         spans_train,\
         spans_entity_train,\
         cluster_ids_train,\
         tokens_train,\
         event_feats_test,\
         entity_feats_test,\
         event_types_test,\
         entity_types_test,\
         ea_maps_test,\
         doc_ids_test,\
         spans_test,\
         spans_entity_test,\
         cluster_ids_test,\
         tokens_test,\
         action_feats_train,\
         object_feats_train,\
         vo_maps_train,\
         action_feats_test,\
         object_feats_test,\
         action_types_test,\
         object_types_test,\
         vo_maps_test,\
         vocab, vocab_entity,\
         action_classes, object_classes


if __name__ == '__main__':
  parser = argparse.ArgumentParser() # formatter=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', '-c', default='../configs/config_graph_dnnhmmdnn_video_m2e2.json')
  args = parser.parse_args()

  config_file = args.config
  config = pyhocon.ConfigFactory.parse_file(config_file)
  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  logging.basicConfig(filename=os.path.join(config['model_path'], 'train.log'))
 
  event_feats_train,\
  entity_feats_train,\
  ea_maps_train,\
  doc_ids_train,\
  spans_train,\
  spans_entity_train,\
  cluster_ids_train,\
  tokens_train,\
  event_feats_test,\
  entity_feats_test,\
  event_types_test,\
  entity_types_test,\
  ea_maps_test,\
  doc_ids_test,\
  spans_test,\
  spans_entity_test,\
  cluster_ids_test,\
  tokens_test,\
  action_feats_train,\
  object_feats_train,\
  vo_maps_train,\
  action_feats_test,\
  object_feats_test,\
  action_types_test,\
  object_types_test,\
  vo_maps_test,\
  vocab, vocab_entity,\
  action_classes, object_classes = load_data(config)
  Ke = len(vocab)
  Ka = len(vocab_entity)
  Kv = len(action_classes)
  Ko = len(object_classes)

  ## Model training
  aligner = GraphMixtureEventAligner(action_feats_train, #+action_feats_test,
                                     object_feats_train, #+object_feats_test,
                                     event_feats_train, #+event_feats_test,
                                     entity_feats_train, #+entity_feats_test,  
                                     vo_maps_train, #+vo_maps_test,
                                     ea_maps_train, #+ea_maps_test,
                                     configs={'n_action_vocab':Kv, 
                                              'n_object_vocab':Ko,
                                              'n_event_vocab':Ke,
                                              'n_entity_vocab':Ka})
  aligner.trainEM(15, os.path.join(config['model_path'], 'mixture')) 
  aligner.align(action_feats_test,
                object_feats_test,
                event_feats_test,
                entity_feats_test,
                action_types_test,
                object_types_test,
                event_types_test,
                entity_types_test,
                vo_maps_test,
                ea_maps_test,
                out_prefix=os.path.join(config['model_path'],
                                        'alignment'))
  
  ## Test and evaluation
  conll_eval = CoNLLEvaluation()

  alignments, _ = aligner.align_sents(action_feats_test,
                                      object_feats_test,
                                      event_feats_test,
                                      entity_feats_test,
                                      vo_maps_test,
                                      ea_maps_test)
  pred_labels = [torch.LongTensor(to_pairwise(a)) for a in alignments if a.shape[0] > 1]
  gold_labels = [torch.LongTensor(to_pairwise(c)) for c in cluster_ids_test if c.shape[0] > 1]
  pred_labels = torch.cat(pred_labels)
  gold_labels = torch.cat(gold_labels)
  
  # Compute pairwise scores
  pairwise_eval = Evaluation(pred_labels, gold_labels)  
  print(f'Pairwise - Precision: {pairwise_eval.get_precision()}, Recall: {pairwise_eval.get_recall()}, F1: {pairwise_eval.get_f1()}')
  logger.info(f'Pairwise precision: {pairwise_eval.get_precision()}, recall: {pairwise_eval.get_recall()}, F1: {pairwise_eval.get_f1()}')
  
  # Compute CoNLL scores and save readable predictions
  f_out = open(os.path.join(config['model_path'], 'prediction.readable'), 'w')
  for doc_id, token, span, alignment, cluster_id in zip(doc_ids_test, tokens_test, spans_test, alignments, cluster_ids_test):
    antecedent = to_antecedents(alignment)
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

  aligner.retrieve(action_feats_test, 
                   object_feats_test, 
                   event_feats_test,
                   entity_feats_test,
                   vo_maps_test,
                   ea_maps_test,
                   os.path.join(config['model_path'], 'retrieval'))
