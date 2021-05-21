import nltk
import numpy as np
import os
import json
import logging
import torch
import codecs
from nltk.stem import WordNetLemmatizer
import itertools
import argparse
import pyhocon
import random
import editdistance
from copy import deepcopy
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from evaluator import Evaluation, CoNLLEvaluation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(2)
random.seed(2)

NULL = '###NEW###'
EPS = 1e-100
PLURAL_PRON = ['they', 'them', 'those', 'these', 'we', 'us', 'their', 'our']
SINGULAR = 'singular'
PLURAL = 'plural'
PROPER = 'proper'
NOMINAL = 'nominal'
PRON = 'pronoun'
MODE_V = 'visual'
MODE_S = 'semantic'
MODE_D = 'discourse' 

class AmodalSMTEventCoreferencer:
  def __init__(self, event_features, action_features, config):
    '''
    Amodal unsupervised coreference system inspired by
    ``Unsupervised Ranking Model for Entity Coreference Resolution``, 
    X. Ma, Z. Liu and E. Hovy, NAACL, 2016.
    
    :param event_features: list of dict of {[feature id]: [feature value]}
    :param action_features: list of array of shape (num. of actions, embedding dim.)
    '''
    self.config = config
    self.text_modes = [MODE_S, MODE_D]
    self.modes = [MODE_S, MODE_D, MODE_V]

    self.e_feats_train = event_features
    self.v_feats_train = action_features
    self.e_null = {'head_lemma': NULL}

    self.ev_counts = dict()
    self.ee_counts = dict()
    self.centroids = None
    self.vars = None
    self.P_ve = dict()
    self.P_ee = dict()
    self.Kv = config.get('Kv', 4)
    self.vocab = self.get_vocab(event_features)
    self.modes_sents = self.get_modes(event_features)
    self.P_ij = {mode: dict() for mode in self.text_modes} 

    self.initialize()
    logging.info(f'Number of documents = {len(self.e_feats_train)}, vocab.size = {len(self.vocab)}')
    
  def get_vocab(self, event_feats):
    vocab = {mode: {NULL: 0} for mode in self.modes}
    for e_feat in event_feats:
      for e_idx, e in enumerate(e_feat):
        for mode in self.modes:
          token = self.get_mode_feature(e, mode)
          if not token in vocab[mode]:
            vocab[mode][token] = 1
          else:
            vocab[mode][token] += 1
    return vocab 
  
  def get_modes(self, event_feats):
    modes_sents = []
    for e_feat in event_feats:
      modes_sent = []
      for e_idx, e in enumerate(e_feat):
        mode = MODE_V # XXX
        '''
        for a_idx, a in enumerate(e_feat[:e_idx]):
          if self.is_match(a, e, MODE_S):
            mode = MODE_S
            break
          elif self.is_match(a, e, MODE_D):
            mode = MODE_D
            break
        if mode == MODE_V and not e['is_visual']:
          mode = MODE_S
        '''
        modes_sent.append(mode)
      modes_sents.append(modes_sent)
    return modes_sents

  def get_mode_feature(self, e, mode):
    if mode in [MODE_S, MODE_V]:
      return e['head_lemma']
    if mode == MODE_D:
      return '__'.join([e['mention_type'], e['number']])

  def initialize(self):
    # Initialize event-event translation probs
    self.P_ee = {mode: dict() for mode in self.modes}
    for mode in self.text_modes:
      for v in self.vocab[mode]:
        self.P_ee[mode][v] = {v2: 1. / (len(self.vocab[mode]) - 1) for v2 in self.vocab[mode] if v2 != NULL} 
    self.P_ee[MODE_V][NULL] = {v: 1. / (len(self.vocab[MODE_V]) - 1) for v in self.vocab[MODE_V] if v != NULL}

    # Initialize position probs
    for mode in self.modes:
      self.P_ij[mode] = dict()
      for e_feat in self.e_feats_train:
        for e_idx, e in enumerate(e_feat):
          e_sent_id = e['sentence_id']
          if not e_sent_id in self.P_ij[mode]:
            self.P_ij[mode][e_sent_id] = 1.
          for a_idx, a in enumerate(e_feat[:e_idx]):
            a_sent_id = a['sentence_id']
            if not e_sent_id-a_sent_id in self.P_ij[mode]:
              self.P_ij[mode][e_sent_id-a_sent_id] = 1.

      norm_factor = sum(self.P_ij[mode].values())
      for d in self.P_ij[mode]:
        self.P_ij[mode][d] /= norm_factor

    # Initialize action-event translation probs
    for k in range(self.Kv):
      self.P_ve[k] = {v: 1. / (len(self.vocab[MODE_V]) - 1) for v in self.vocab[MODE_V] if v != NULL} 

    # Initialize action cluster centroids
    X = np.concatenate(self.v_feats_train)
    kmeans = KMeans(n_clusters=self.Kv).fit(X)

    self.centroids = deepcopy(kmeans.cluster_centers_)
    y = kmeans.predict(X)
    self.vars = 0.01 * np.ones(self.Kv)
    # XXX for k in range(self.Kv):
    #   self.vars[k] = np.var(X[(y == k).nonzero()[0]]) * X.shape[1] / 3

  def is_str_match(self, e1, e2):
    if e1['tokens'] == e2['tokens']:
      return True
    if e1['head_lemma'] == e2['head_lemma']:
      return True
    if (e1['tokens'] in e2['tokens']) or (e2['tokens'] in e1['tokens']): 
      return True
    ned = editdistance.eval(e1['tokens'], e2['tokens']) / max(len(e1['tokens']), len(e2['tokens']))
    if ned <= 0.5:
      return True
    ned_lemma = editdistance.eval(e1['head_lemma'], e2['head_lemma']) / max(len(e1['head_lemma']), len(e2['head_lemma']))
    if ned_lemma <= 0.5:
      return True
    return False
 
  def is_number_match(self, e1, e2):
    return e1['number'] == e2['number']

  def is_argument_match(self, e1, e2):
    a1_by_types = dict()
    for i, a1 in enumerate(e1['arguments']):
      if not a1['entity_type'] in a1_by_types:
        a1_by_types[a1['entity_type']] = [i]
      else:
        a1_by_types[a1['entity_type']].append(i)
    
    a2_by_types = dict()
    for i, a2 in enumerate(e2['arguments']):
      if not a2['entity_type'] in a2_by_types:
        a2_by_types[a2['entity_type']] = [i]
      else:
        a2_by_types[a2['entity_type']].append(i)

    if len(set(a2_by_types.keys()).intersection(set(a1_by_types.keys()))) == 0:
      return True

    for a_type in a2_by_types:
      if not a_type in a1_by_types:
        continue
      for a2_idx in a2_by_types[a_type]:
        a2 = e2['arguments'][a2_idx]
        a1s = [e1['arguments'][i] for i in a1_by_types[a_type]]

        if a_type == 'Person':
          matched = False
          for a1 in a1s:
            if self.is_str_match(a1, a2):
              matched = True
              break
            elif cosine_similarity(a1['entity_embedding'], a2['entity_embedding']) > 0.4:
              matched = True
          if not matched:
            return False
             
    return True

  def is_match(self, e1, e2, mode):
    if e1['head_lemma'] != NULL:
      if e1['event_type'] != e2['event_type']:
        return False
    if mode == MODE_S:
      if e1['head_lemma'] == NULL:
        if e2['pos_tag'] in ['PRP', 'PRP$']:
          return False
        return True
      v1 = e1['trigger_embedding']
      v2 = e2['trigger_embedding']
      if e1['is_visual'] != e2['is_visual']:
        return False
      if cosine_similarity(v1, v2) <= 0.5:
        return False 
      if not self.is_number_match(e1, e2):
        return False
    elif mode == MODE_D:
      if e1['head_lemma'] == NULL:
        if e2['pos_tag'] in ['PRP', 'PRP$']:
          return False
      if (e1['mention_type'], e2['mention_type']) != (NOMINAL, PRON):
        return False
    elif mode == MODE_V:
      if e1['head_lemma'] == NULL:
        return True
      if not e1['is_visual'] or not e2['is_visual']:
        return False
      if not self.is_number_match(e1, e2):
        return False
      e1_token = self.get_mode_feature(e1, MODE_V)
      if not e1_token in self.P_ve[0]:
        return False
    return True  

  def compute_alignment_counts(self):
    ev_counts = []
    ee_counts = []
    for e_feat, v_feat, modes_sent in zip(self.e_feats_train, self.v_feats_train, self.modes_sents):
      C_ee = dict()
      C_ev = dict()
      L = v_feat.shape[0]
      v_prob = self.compute_cluster_prob(v_feat)

      for e_idx, (e, mode) in enumerate(zip(e_feat, modes_sent)):
        token = self.get_mode_feature(e, mode)
        e_sent_id = e['sentence_id']

        # Compute event-event alignment counts
        if mode in self.text_modes:
          if not mode in C_ee:
            C_ee[mode] = dict()
          C_ee[mode][e_idx] = dict()
          if self.is_match(self.e_null, e, mode):
            C_ee[mode][e_idx][0] = self.P_ij[mode][e_sent_id] * self.P_ee[mode][NULL][token] 
          for a_idx, a in enumerate(e_feat[:e_idx]):
            if self.is_match(a, e, mode):
              a_token = self.get_mode_feature(a, mode)  
              a_sent_id = a['sentence_id']
              d = e_sent_id-a_sent_id
              C_ee[mode][e_idx][a_idx+1] = self.P_ij[mode][d] * self.P_ee[mode][a_token][token]
          norm_factor = sum(C_ee[mode][e_idx].values())
          for a_idx in C_ee[mode][e_idx]:
            C_ee[mode][e_idx][a_idx] /= max(norm_factor, EPS)

        # Compute event-action alignment counts
        elif mode == MODE_V:
          C_ev[e_idx] = dict()
          if not MODE_V in C_ee:
            C_ee[MODE_V] = dict()
          C_ee[MODE_V][e_idx] = dict()
          C_ee[MODE_V][e_idx][0] = self.P_ij[MODE_V][e_sent_id] * self.P_ee[MODE_V][NULL][token] 

          e_prob = self.action_event_prob(e)
          for v_idx, v in enumerate(v_feat):
            C_ev[e_idx][v_idx] = self.P_ij[MODE_V][e_sent_id] * v_prob[v_idx] * e_prob 

          norm_factor = C_ee[MODE_V][e_idx][0]
          norm_factor += sum(C_ev[e_idx][v_idx].sum() for v_idx in C_ev[e_idx])          
          
          C_ee[MODE_V][e_idx][0] /= max(norm_factor, EPS)
          for v_idx in C_ev[e_idx]:
            C_ev[e_idx][v_idx] /= max(norm_factor, EPS)
            C_ev[e_idx][v_idx] = C_ev[e_idx][v_idx].tolist()  
      ee_counts.append(C_ee)
      ev_counts.append(C_ev)
    return ee_counts, ev_counts 

  def update_translation_probs(self):
    P_ee = {mode: dict() for mode in self.modes}
    P_ve = {k:dict() for k in range(self.Kv)}
    for e_feat, v_feat, ee_count, ev_count, modes_sent in zip(self.e_feats_train, self.v_feats_train, self.ee_counts, self.ev_counts, self.modes_sents):
      for mode in ee_count:
        for e_idx in ee_count[mode]:
          token = self.get_mode_feature(e_feat[e_idx], mode)
          for a_idx in ee_count[mode][e_idx]:
            if a_idx == 0:
              if not NULL in P_ee[mode]:
                P_ee[mode][NULL] = dict()
              
              if not token in P_ee[mode][NULL]:
                P_ee[mode][NULL][token] = ee_count[mode][e_idx][a_idx]
              else:
                P_ee[mode][NULL][token] += ee_count[mode][e_idx][a_idx]
            else:
              a_token = self.get_mode_feature(e_feat[a_idx-1], mode)
              if not a_token in P_ee[mode]:
                P_ee[mode][a_token] = dict()

              if not token in P_ee[mode][a_token]:
                P_ee[mode][a_token][token] = ee_count[mode][e_idx][a_idx]
              else:
                P_ee[mode][a_token][token] += ee_count[mode][e_idx][a_idx]
        
      for e_idx in ev_count:
        token = self.get_mode_feature(e_feat[e_idx], MODE_V)
        for v_idx in ev_count[e_idx]:
          for k in range(self.Kv):
            if not token in P_ve[k]:
              P_ve[k][token] = ev_count[e_idx][v_idx][k]
            else:
              P_ve[k][token] += ev_count[e_idx][v_idx][k]

    # Normalize
    for mode in P_ee:
      for a in P_ee[mode]:
        norm_factor = sum(P_ee[mode][a].values())
        for e in P_ee[mode][a]:
          P_ee[mode][a][e] /= max(norm_factor, EPS)
    
    for v in P_ve:
      norm_factor = sum(P_ve[v].values())
      for e in P_ve[v]:
        P_ve[v][e] /= max(norm_factor, EPS)
    return P_ee, P_ve

  def update_position_probs(self):
    P_ij = dict()
    for e_feat, count, modes_sent in zip(self.e_feats_train, self.ee_counts, self.modes_sents):
      for mode in count:
        if mode == MODE_D:
          for e_idx in count[mode]:
            e_sent_id = e_feat[e_idx]['sentence_id']
            for a_idx in count[mode][e_idx]:
              a_sent_id = e_feat[a_idx-1]['sentence_id']
              if not e_sent_id - a_sent_id in P_ij:
                P_ij[e_sent_id - a_sent_id] = count[mode][e_idx][a_idx]
              else:
                P_ij[e_sent_id - a_sent_id] += count[mode][e_idx][a_idx]

    norm_factor = sum(P_ij.values())
    for d in P_ij:
      P_ij[d] /= max(norm_factor, EPS)

    return P_ij

  def log_likelihood(self):
    ll = 0.
    for e_feat, v_feat, modes_sent in zip(self.e_feats_train, self.v_feats_train, self.modes_sents):
      v_prob = self.compute_cluster_prob(v_feat)
      for e_idx, (e, mode) in enumerate(zip(e_feat, modes_sent)):
        e_token = self.get_mode_feature(e, mode)
        e_sent_id = e['sentence_id']
        
        if self.is_match(self.e_null, e, mode):
          probs = [self.P_ij[mode][e_sent_id] * self.P_ee[mode][NULL][e_token]]
        else:
          probs = [0]

        if mode in self.text_modes:
          for a_idx, a in enumerate(e_feat[:e_idx]):
            a_token = self.get_mode_feature(a, mode)
            a_sent_id = a['sentence_id']
            d = e_sent_id - a_sent_id
            if self.is_match(a, e, mode):
              probs.append(self.P_ij[mode][d] * self.P_ee[mode][a_token][e_token])  
        else:
          e_prob = self.action_event_prob(e)
          probs.extend((self.P_ij[mode][e_sent_id] * (v_prob @ e_prob)).tolist()) 
        ll += np.log(np.maximum(np.mean(probs), EPS))
    return ll
           
  def train(self, n_epochs=10):
    for epoch in range(n_epochs):
      self.ee_counts, self.ev_counts = self.compute_alignment_counts()
      self.P_ee, self.P_ve = self.update_translation_probs()
      self.P_ij[MODE_D] = self.update_position_probs()
                 
      json.dump(self.P_ee, open(os.path.join(self.config['model_path'], 'ee_translation_probs.json'), 'w'), indent=2)
      json.dump(self.P_ve, open(os.path.join(self.config['model_path'], 've_translation_probs.json'), 'w'), indent=2)
      json.dump(self.P_ij, open(os.path.join(self.config['model_path'], 'position_probs.json'), 'w'), indent=2)

      logging.info('Epoch {}, log likelihood = {:.3f}'.format(epoch, self.log_likelihood()))           
      print('Epoch {}, log likelihood = {:.3f}'.format(epoch, self.log_likelihood()))
 
  def compute_cluster_prob(self, v_feat):
    # (num. of actions, num. of clusters)
    logit = - (v_feat**2 / 2.).sum(axis=-1, keepdims=True)\
            + v_feat @ self.centroids.T\
            - (self.centroids**2 / 2.).sum(axis=-1)  
    logit /= self.vars
    log_prob = logit - logsumexp(logit, axis=1)[:, np.newaxis]
    return np.exp(log_prob)

  def action_event_prob(self, e):
    e_token = self.get_mode_feature(e, MODE_V)
    return np.asarray([self.P_ve[k][e_token] for k in range(self.Kv)])

  def predict_antecedents(self, event_features, action_features):
    antecedents = []
    cluster_ids = []
    scores_all = []
    modes_sents = self.get_modes(event_features)

    n_cluster = 0
    for e_feat, v_feat, modes_sent in zip(event_features, action_features, modes_sents):
      v_prob = self.compute_cluster_prob(v_feat)
      antecedent = [-1]*len(e_feat)
      cluster_id = [0]*len(e_feat)
      scores_all.append([])
      L = v_feat.shape[0]
      for e_idx, (e, mode) in enumerate(zip(e_feat, modes_sent)):
        e_token = self.get_mode_feature(e, mode)
        e_sent_id = e['sentence_id']
        if self.is_match(self.e_null, e, mode):
          null_prob = self.P_ee[mode][NULL][e_token] 
          if mode in self.text_modes:
            scores = [self.P_ij[mode][e_sent_id] * null_prob]
          else:
            e_prob = self.action_event_prob(e)
            ve_prob = np.append(v_prob @ e_prob, [null_prob])
            ve_prob /= ve_prob.sum()
            scores = [ve_prob[-1]]
        else:
          scores = [0]

        for a_idx, a in enumerate(e_feat[:e_idx]):
          a_sent_id = a['sentence_id']
          d = e_sent_id - a_sent_id
          if self.is_match(a, e, mode): # XXX and self.is_argument_match(a, e):
            if mode in self.text_modes:
              a_token = self.get_mode_feature(a, mode)
              scores.append(self.P_ij[mode][d] * self.P_ee[mode][a_token][e_token])
            else:
              e_prob = self.action_event_prob(e)
              a_prob = self.action_event_prob(a)
              ve_prob = np.append(v_prob @ e_prob, [null_prob])
              va_prob = np.append(v_prob @ a_prob, [null_prob])
              ve_prob /= ve_prob.sum()
              va_prob /= va_prob.sum()
              scores.append(ve_prob @ va_prob) 
          else:
            scores.append(0)
       
        scores_all[-1].append(scores)
        scores = np.asarray(scores)

        antecedent[e_idx] = int(np.argmax(scores)) - 1
        # If antecedent idx == -1, the mention belongs to a new cluster; 
        # if antecedent idx < -1, the mention belongs to a visual cluster, 
        # need to check all previous antecedents to decide its cluster id; 
        if antecedent[e_idx] == -1: 
          n_cluster += 1
          cluster_id[e_idx] = n_cluster
        else:
          cluster_id[e_idx] = cluster_id[antecedent[e_idx]]
      antecedents.append(antecedent)
      cluster_ids.append(cluster_id)
    return antecedents, cluster_ids, scores_all

def cosine_similarity(v1, v2):
  return abs(v1 @ v2) / np.maximum(EPS, np.linalg.norm(v1) * np.linalg.norm(v2))

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

def get_number_and_mention_type(entity):
  token = entity['head_lemma']
  pos_tag = entity['pos_tag']
  if pos_tag in ['PRP', 'PRP$']:
    entity['mention_type'] = PRON
    if token in PLURAL_PRON:
      entity['number'] = PLURAL
    else:
      entity['number'] = SINGULAR
  elif pos_tag in ['NNP', 'NNPS']:
    entity['mention_type'] = PROPER
    if pos_tag[-1] == 'S':
      entity['number'] = PLURAL
    else:
      entity['number'] = SINGULAR
  elif pos_tag in ['NN', 'NNS']:
    entity['mention_type'] = NOMINAL
    if pos_tag[-1] == 'S':
      entity['number'] = PLURAL
    else:
      entity['number'] = SINGULAR
  elif pos_tag == 'CD':
    entity['mention_type'] = NOMINAL
    entity['number'] = token
  else:
    entity['mention_type'] = NOMINAL
    entity['number'] = SINGULAR
  return entity 

def get_proper_mention(entity, label_dict):
  cluster_id = entity['cluster_id']
  if cluster_id == 0:
    return entity
  mention_type = entity['mention_type']
  found = False
  for span, m in label_dict.items():
    if (m['cluster_id'] == cluster_id) and (m['mention_type'] == PROPER):
      found = True
      return m

  if not found and (mention_type == PRON):
    for span, m in label_dict.items():
      if (m['cluster_id'] == cluster_id) and (m['mention_type'] == NOMINAL):
        return m
  return entity

def load_event_features(config, split, action_labels=None):
  lemmatizer = WordNetLemmatizer()
  event_feature_types = config['feature_types']
  event_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_events.json'), 'r', 'utf-8'))
  entity_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_entities.json'), 'r', 'utf-8')) 

  ontology_map = json.load(open(os.path.join(config['data_folder'], '../ontology_mapping.json')))
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}.json')))
  event_docs_embs = np.load(os.path.join(config['data_folder'], f'{split}_events_with_arguments_glove_embeddings.npz')) 
  entity_docs_embs = np.load(os.path.join(config['data_folder'], f'{split}_entities_glove_embeddings.npz'))
  doc_to_event_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in event_docs_embs}
  doc_to_entity_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in entity_docs_embs}

  entity_label_dicts = dict()
  event_label_dicts = dict()
  event_feats = []
  doc_ids = []
  spans_all = []
  cluster_ids_all = []
  tokens_all = [] 

  for m in entity_mentions:
    if not m['doc_id'] in entity_label_dicts:
      entity_label_dicts[m['doc_id']] = dict()
    span = (min(m['tokens_ids']), max(m['tokens_ids'])) 
    m = get_number_and_mention_type(m)
    entity_label_dicts[m['doc_id']][span] = deepcopy(m)

  for doc_id in sorted(entity_label_dicts):
    for span_idx, span in enumerate(sorted(entity_label_dicts[doc_id])):
      entity_doc_embs = entity_docs_embs[doc_to_entity_feat[doc_id]]
      entity_label_dicts[doc_id][span]['entity_embedding'] = entity_doc_embs[span_idx, :300]  

  # Replace pronouns with its coreferent proper mention
  new_entity_label_dicts = dict() 
  for doc_id in entity_label_dicts:
    new_entity_dict = dict()
    for span, m in entity_label_dicts[doc_id].items():
      if m['mention_type'] in [PRON, NOMINAL]:
        new_entity_dict[span] = get_proper_mention(m, entity_label_dicts[doc_id])
      else:
        new_entity_dict[span] = deepcopy(m)
    new_entity_label_dicts[doc_id] = deepcopy(new_entity_dict)
  entity_label_dicts = deepcopy(new_entity_label_dicts)

  for m in event_mentions:
    if not m['doc_id'] in event_label_dicts:
      event_label_dicts[m['doc_id']] = dict()
    token = lemmatizer.lemmatize(m['tokens'].lower(), pos='v')
    span = (min(m['tokens_ids']), max(m['tokens_ids']))
    event_label_dicts[m['doc_id']][span] = {'token_id': token,
                                            'cluster_id': m['cluster_id']}

    for feat_type in event_feature_types:
      event_label_dicts[m['doc_id']][span][feat_type] = m[feat_type]

  for feat_idx, doc_id in enumerate(sorted(event_label_dicts)): # XXX
    event_doc_embs = event_docs_embs[doc_to_event_feat[doc_id]]
    event_label_dict = event_label_dicts[doc_id]

    spans = sorted(event_label_dict)
    events = []
    for span_idx, span in enumerate(spans):
      event = {feat_type: event_label_dict[span][feat_type] for feat_type in event_feature_types}
      event['trigger_embedding'] = event_doc_embs[span_idx, :300]
      event = get_number_and_mention_type(event)

      for a_idx, a in enumerate(event['arguments']):
        a_span = (a['start'], a['end'])
        event['arguments'][a_idx] = deepcopy(entity_label_dicts[doc_id][a_span]) 

      if 'event_type' in event:
        match_action_types = ontology_map.get(event['event_type'], [])
         
        ''' XXX
        if action_labels is not None:
          action_label = action_labels[feat_idx]
          if len(set(match_action_types).intersection(set(action_label))) > 0:
            event['is_visual'] = True
          else:
            event['is_visual'] = False
        '''
        if len(match_action_types) > 0:
          event['is_visual'] = True
        else:
          event['is_visual'] = False
        event['is_visual'] = True
        
      events.append(event)  
    cluster_ids = [event_label_dict[span]['cluster_id'] for span in spans]
    event_feats.append(events)
    doc_ids.append(doc_id)
    spans_all.append(spans)
    cluster_ids_all.append(np.asarray(cluster_ids))
    tokens_all.append([t[2] for t in doc_train[doc_id]])

  return event_feats,\
         doc_ids,\
         spans_all,\
         cluster_ids_all,\
         tokens_all

def load_visual_features(config, split):
  ontology = json.load(open(os.path.join(config['data_folder'], '../ontology.json')))
  action_classes = ontology['event'] 
  event_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_events.json'), 'r', 'utf-8'))
  label_dicts = dict()
  for m in event_mentions:
    if not m['doc_id'] in label_dicts:
      label_dicts[m['doc_id']] = dict()
      span = (min(m['tokens_ids']), max(m['tokens_ids']))
      label_dicts[m['doc_id']][span] = m['cluster_id']
  action_npz = np.load(os.path.join(config['data_folder'], f'{split}_mmaction_feat.npz')) # f'{split}_mmaction_feat.npz')) # f'{split}_original_vid_3d.npz')) # f'{split}_mmaction_feat.npz')) # XXX f'{split}_events_event_type_labels.npz'
    
  action_label_path = os.path.join(config['data_folder'], f'{split}_mmaction_event_feat_labels_average.npz')
  if os.path.exists(action_label_path):
    action_label_npz = np.load(action_label_path)
  else:
    action_label_npz = None

  doc_to_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in sorted(action_npz, key=lambda x:int(x.split('_')[-1]))}
  action_feats = [action_npz[doc_to_feat[doc_id]] for doc_id in sorted(label_dicts)] 
  
  if action_label_npz is not None:
    doc_to_label = {'_'.join(label_id.split('_')[:-1]):label_id for label_id in sorted(action_label_npz, key=lambda x:int(x.split('_')[-1]))}
    action_labels_onehot = [action_label_npz[doc_to_label[doc_id]] for doc_id in sorted(label_dicts)]
    action_labels = [[action_classes[y] for y in np.argmax(action_label, axis=-1)] for action_label in action_labels_onehot]
  else:
    action_labels = None
  return action_feats, action_labels

def load_data(config):
  event_feats_train = []
  action_feats_train = []
  action_labels_train = []
  doc_ids_train = []
  spans_train = []
  cluster_ids_train = []
  tokens_train = [] 
  ontology_map = json.load(open(os.path.join(config['data_folder'], '../ontology_mapping.json')))
  event_stoi = {k:i for i, k in enumerate(ontology_map)}
  n_types = len(event_stoi)

  for split in config['splits']['train']:
    cur_action_feats_train, cur_action_labels_train = load_visual_features(config, split=split)
    cur_event_feats_train,\
    cur_doc_ids_train,\
    cur_spans_train,\
    cur_cluster_ids_train,\
    cur_tokens_train = load_event_features(config, split=split, action_labels=cur_action_labels_train)
    
    # cur_action_feats_train = [[np.eye(n_types)[event_stoi[e['event_type']]] for e in e_feat if e['is_visual']] for e_feat in cur_event_feats_train] # XXX 
    # cur_action_feats_train = [np.stack(a_feat) if len(a_feat) > 0 else np.zeros((1,n_types)) for a_feat in cur_action_feats_train]
    action_feats_train.extend(cur_action_feats_train)
    if cur_action_labels_train is not None:
      action_labels_train.extend(cur_action_labels_train)
    else:
      action_labels_train.extend([[0] for _ in range(len(cur_action_feats_train))])
    event_feats_train.extend(cur_event_feats_train)
    doc_ids_train.extend(cur_doc_ids_train)
    spans_train.extend(cur_spans_train)
    cluster_ids_train.extend(cur_cluster_ids_train)
    tokens_train.extend(cur_tokens_train)
  print(f'Number of training examples: {len(event_feats_train)}')

  event_feats_test = []
  action_feats_test = []
  action_labels_test = []
  doc_ids_test = []
  spans_test = []
  cluster_ids_test = []
  tokens_test = []
  for split in config['splits']['test']:
    cur_action_feats_test, cur_action_labels_test = load_visual_features(config, split=split)
    cur_event_feats_test,\
    cur_doc_ids_test,\
    cur_spans_test,\
    cur_cluster_ids_test,\
    cur_tokens_test = load_event_features(config, split=split, action_labels=cur_action_labels_test)

    # cur_action_feats_test = [[np.eye(n_types)[event_stoi[e['event_type']]] for e in e_feat if e['is_visual']] for e_feat in cur_event_feats_test] # XXX
    action_feats_test.extend(cur_action_feats_test)
    if cur_action_labels_test is not None:
      action_labels_test.extend(cur_action_labels_test)
    else:
      action_labels_test.extend([[0] for _ in range(len(cur_action_feats_test))])
    event_feats_test.extend(cur_event_feats_test)
    doc_ids_test.extend(cur_doc_ids_test)
    spans_test.extend(cur_spans_test)
    cluster_ids_test.extend(cur_cluster_ids_test)
    tokens_test.extend(cur_tokens_test)
  print(f'Number of test examples: {len(event_feats_test)}')
  
  return event_feats_train,\
         action_feats_train,\
         action_labels_train,\
         doc_ids_train,\
         spans_train,\
         cluster_ids_train,\
         tokens_train,\
         event_feats_test,\
         action_feats_test,\
         action_labels_test,\
         doc_ids_test,\
         spans_test,\
         cluster_ids_test,\
         tokens_test
         
def plot_attention(prediction, e_feats, v_labels, out_prefix):
  fig, ax = plt.subplots(figsize=(7, 10))
  scores = prediction['score']
  num_events = len(e_feats)
  num_actions = len(v_labels)
  e_tokens = [e['head_lemma'] for e in e_feats]
  score_mat = []
  for score in scores:
    if len(score) < num_events + num_actions + 1:
      gap = num_events + num_actions + 1 - len(score)
      score.extend([0]*gap)
      score_mat.append(score)

  score_mat = np.asarray(score_mat).T
  score_mat /= np.maximum(score_mat.sum(0), EPS) 
  si = np.arange(num_events+1)
  ti = np.arange(num_events+num_actions+2)
  S, T = np.meshgrid(si, ti)
  plt.pcolormesh(S, T, score_mat)
  for i in range(num_events):
    for j in range(num_events+num_actions+1):
      plt.text(i, j, round(score_mat[j, i], 2), color='orange')
  ax.set_xticks(si[1:]-0.5)
  ax.set_yticks(ti[1:]-0.5)
  ax.set_xticklabels(e_tokens)
  ax.set_yticklabels(v_labels+[NULL]+e_tokens) 
  plt.xticks(rotation=45)
  plt.colorbar()
  plt.savefig(out_prefix+'.png')
  plt.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', '-c', default='../configs/config_amodal_smt_video_m2e2.json')
  args = parser.parse_args()

  config_file = args.config
  config = pyhocon.ConfigFactory.parse_file(config_file)
  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  logging.basicConfig(filename=os.path.join(config['model_path'], 'train.log'), level=logging.DEBUG)

  event_feats_train,\
  action_feats_train,\
  action_labels_train,\
  doc_ids_train,\
  spans_train,\
  cluster_ids_train,\
  tokens_train,\
  event_feats_test,\
  action_feats_test,\
  action_labels_test,\
  doc_ids_test,\
  spans_test,\
  cluster_ids_test,\
  tokens_test = load_data(config)

  ## Model training
  aligner = AmodalSMTEventCoreferencer(event_feats_train+event_feats_test, action_feats_train+action_feats_test, config)
  aligner.train(5)
  antecedents, cluster_ids_all, scores_all = aligner.predict_antecedents(event_feats_test, action_feats_test)

  predictions = [{'doc_id': doc_id,
                  'antecedent': antecedent,
                  'score': score}  
                  for doc_id, antecedent, score in
                       zip(doc_ids_test, antecedents, scores_all)]
  json.dump(predictions, open(os.path.join(config['model_path'], 'predictions.json'), 'w'), indent=2)

  ## Test and Evaluation
  pred_cluster_ids = [np.asarray(cluster_ids) for cluster_ids in cluster_ids_all]
  pred_labels = [torch.LongTensor(to_pairwise(a)) for a in pred_cluster_ids if a.shape[0] > 1]
  gold_labels = [torch.LongTensor(to_pairwise(c)) for c in cluster_ids_test if c.shape[0] > 1]
  pred_labels = torch.cat(pred_labels)
  gold_labels = torch.cat(gold_labels)

  # Compute pairwise scores
  pairwise_eval = Evaluation(pred_labels, gold_labels)  
  print(f'Pairwise - Precision: {pairwise_eval.get_precision()}, Recall: {pairwise_eval.get_recall()}, F1: {pairwise_eval.get_f1()}')
  logging.info(f'Pairwise precision: {pairwise_eval.get_precision()}, recall: {pairwise_eval.get_recall()}, F1: {pairwise_eval.get_f1()}')
  
  # Compute CoNLL scores and save readable predictions
  conll_eval = CoNLLEvaluation()
  f_out = open(os.path.join(config['model_path'], 'prediction.readable'), 'w')
  select_ids = [] 
  # ['92PLcoWtn0Q', '9tx72NIbwh8', 'AohILHV6i8Q', 'GLOGR0UsBtk', 
  # 'LHIbc7koTUE', 'PaVqCYxGzp0', 'SvrpxITQ3Pk', 'dY_hkbVQA20', 
  # 'eaW-mv9IKOs', 'f3plTR1Dcew', 'fDm7S-pjpOo', 'fsYMznJdCok']
  for doc_idx, (doc_id, token, span, antecedent, pred_cluster_id, gold_cluster_id) in enumerate(zip(doc_ids_test, tokens_test, spans_test, antecedents, pred_cluster_ids, cluster_ids_test)):
    pred_clusters, gold_clusters = conll_eval(torch.LongTensor(span),
                                              torch.LongTensor(antecedent),
                                              torch.LongTensor(span),
                                              torch.LongTensor(gold_cluster_id))
    # Plot attention maps for selected ids
    if doc_id in select_ids:
      plot_attention(predictions[doc_idx], event_feats_test[doc_idx], action_labels_test[doc_idx], out_prefix=os.path.join(config['model_path'], doc_id))
     
    arguments = {(min(e['tokens_ids']), max(e['tokens_ids'])):e['arguments'] for e in event_feats_test[doc_idx]} 
    pred_clusters_str, gold_clusters_str = conll_eval.make_output_readable(pred_clusters, gold_clusters, token, arguments) 
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
  logging.info('MUC - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'Bcubed - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CEAFe - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CoNLL - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(*conll_metrics))
