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
from copy import deepcopy
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from evaluator import Evaluation, CoNLLEvaluation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from amodal_smt_event_coreferencer import AmodalSMTEventCoreferencer
from text_only_smt_entity_coreferencer import SMTEntityCoreferencer 
np.random.seed(2)
random.seed(2)

NULL = '###NEW###'
UNK = '###UNK###'
EPS = 1e-100
PLURAL_PRON = ['they', 'them', 'those', 'these', 'we', 'us', 'their', 'our']
SINGULAR = 'singular'
PLURAL = 'plural'
PROPER = 'proper'
NOMINAL = 'nominal'
PRON = 'pronoun'

class AmodalSMTJointCoreferencer:
  def __init__(self, event_features, entity_features, action_features, config):
    '''
    Amodal unsupervised coreference system inspired by
    ``Unsupervised Ranking Model for Entity Coreference Resolution``, 
    X. Ma, Z. Liu and E. Hovy, NAACL, 2016.
    
    :param event_features: list of dict of {[feature id]: [feature value]}
    :param action_features: list of array of shape (num. of actions, embedding dim.)
    '''
    self.config = config
    argument_features = self.get_argument_features(event_features)
    self.e_feats_train = event_features
    self.a_feats_train = argument_features # entity_features
    self.v_feats_train = action_features
    self.event_coref_model = AmodalSMTEventCoreferencer(event_features, action_features, config)
    self.entity_coref_model = SMTEntityCoreferencer(entity_features, config)

    self.P_ea = dict()
    self.Kv = config.get('Kv', 4)
    self.vocab = self.get_vocab(event_features)
    self.event_modes_sent, self.entity_modes_sent = self.get_modes(event_features)

    self.initialize()
    logging.info(f'Number of documents = {len(self.e_feats_train)}, vocab.size = {len(self.vocab)}')
    
  def get_argument_features(self, event_features):
    return [[a for e in e_feat for a in e['arguments']] for e_feat in event_features]

  def get_vocab(self, event_feats):
    vocab = {NULL:0}
    for e_feat in event_feats:
      for e_idx, e in enumerate(e_feat):
        if not e['head_lemma'] in vocab:
          vocab[e['head_lemma']] = 1
        else:
          vocab[e['head_lemma']] += 1
    return vocab 
  
  def get_modes(self, event_feats):
    event_modes_sents = [['textual' for _ in e_feat] for e_feat in event_feats] # XXX self.event_coref_model.get_modes(event_feats) 
    argument_feats = self.get_argument_features(event_feats) 
    entity_modes_sents = self.entity_coref_model.get_modes(argument_feats)
    argument_modes_sents = []
    for e_feat, event_modes_sent, entity_modes_sent in zip(event_feats, event_modes_sents, entity_modes_sents):
      entity_idx = 0
      argument_modes_sent = []
      for e in e_feat:
        cur_modes = []
        for a in e['arguments']:
          cur_modes.append(entity_modes_sent[entity_idx])
          entity_idx += 1
        argument_modes_sent.append(cur_modes)
      argument_modes_sents.append(argument_modes_sent)
    return event_modes_sents, argument_modes_sents

  def initialize(self):
    for e_feat in self.e_feats_train:
      for e_idx, e in enumerate(e_feat):
        e_token = e['head_lemma']
        for a in e['arguments']:
          a_token = a['head_lemma']
          a_type = a['entity_type']
          if not a_type in self.P_ea:
            self.P_ea[a_type] = dict()
            self.P_ea[a_type][NULL] = dict()
           
          if not e_token in self.P_ea[a_type]:
            self.P_ea[a_type][e_token] = dict()
            self.P_ea[a_type][e_token][UNK] = 1
          
          if not a_token in self.P_ea[a_type][e_token]:
            self.P_ea[a_type][NULL][a_token] = 1
            self.P_ea[a_type][e_token][a_token] = 1
          else:
            self.P_ea[a_type][NULL][a_token] += 1
            self.P_ea[a_type][e_token][a_token] += 1

    for a_type in self.P_ea:
      for e_token in self.P_ea[a_type]:
        norm_factor = sum(self.P_ea[a_type][e_token].values())
        for a_token in self.P_ea[a_type][e_token]:
          self.P_ea[a_type][e_token][a_token] /= norm_factor

  def has_same_plurality(self, e1, e2):
    if e1.get('word_class', 'VERB') == 'NOUN' and e2.get('word_class', 'VERB') == 'NOUN':
      if (e1['pos_tag'][-1] == 'S' and e2['pos_tag'][-1] != 'S') or (e2['pos_tag'][-1] == 'S' and e1['pos_tag'][-1] != 'S'):
        return False 
    elif e1.get('word_class', 'VERB') == 'NOUN' and e2.get('word_class', 'VERB') == 'VERB':
      if e1['pos_tag'][-1] == 'S':
        return False
    elif e2.get('word_class', 'VERB') == 'NOUN' and e1.get('word_class', 'VERB') == 'VERB':
      if e2['pos_tag'][-1] == 'S':
        return False
    return True
  
  def is_match(self, e1, e2, mode):
    if not self.event_coref_model.is_match(e1, e2, mode):
      return False
    
    first_args = [a1['tokens'] for a1 in e1['arguments']]
    second_args = [a2['tokens'] for a2 in e2['arguments']]
    first_cluster_ids = [a1['cluster_id'] for a1 in e1['arguments']]
    second_cluster_ids = [a2['cluster_id'] for a2 in e2['arguments']]
    
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

    prob = 1.
    for a_type in a2_by_types:
      if not a_type in a1_by_types:
        continue
      for a2_idx in a2_by_types[a_type]:
        a2 = e2['arguments'][a2_idx]
        a1s = [e1['arguments'][i] for i in a1_by_types[a_type]]
        
        
        if (a_type == 'GeopoliticalEntity') and (a2['mention_type'] == PROPER):
          a2_emb = a2['entity_embedding']
          a1_embs = [a1['entity_embedding'] for a1 in a1s if a1['mention_type'] == PROPER]
          if len(a1_embs) > 0:
            score = max([cosine_similarity(a1_emb, a2_emb) for a1_emb in a1_embs])
            if score <= 0.5:
              return False 
        elif a_type == 'Person':
          c2 = a2['cluster_id']
          c1s = [a1['cluster_id'] for a1 in a1s]  
          print(e2['doc_id'], e2['head_lemma'], a2['tokens'], c2, e1['head_lemma'], [a1['tokens'] for a1 in a1s], [a1['cluster_id'] for a1 in a1s]) # XXX
          if (len(c1s) > 0) and not c2 in c1s:
            return False
    return True
       
  def train(self, n_epochs=10):
    self.event_coref_model.train(n_epochs=n_epochs)
    self.entity_coref_model.train(n_epochs=n_epochs)
    _, cluster_ids = self.entity_coref_model.predict_antecedents(self.entity_coref_model.e_feats_train)
    label_dict = dict()
    for cluster_id, entity_feat in zip(cluster_ids, self.entity_coref_model.e_feats_train):
      for c, e in zip(cluster_id, entity_feat):
        doc_id = e['doc_id']
        span = (min(e['tokens_ids']), max(e['tokens_ids']))
      
        if not doc_id in label_dict:
          label_dict[doc_id] = dict()
        label_dict[doc_id][span] = c

    for idx, e_feat in enumerate(self.e_feats_train): 
      for e_idx, e in enumerate(e_feat):
        for a_idx, a in enumerate(e['arguments']):
          doc_id = a['doc_id']
          span = (min(a['tokens_ids']), max(a['tokens_ids']))
          cluster_id = label_dict[doc_id][span]
          self.e_feats_train[idx][e_idx]['arguments'][a_idx]['cluster_id'] = cluster_id

  def compute_cluster_prob(self, v_feat):
    return self.event_coref_model.compute_cluster_prob(v_feat)

  def action_event_prob(self, e):
    P_ve = self.event_coref_model.P_ve
    # P_va = self.entity_coref_model.P_va

    e_token = e['head_lemma']
    e_prob = np.asarray([P_ve[k][e_token] for k in range(self.Kv)])
    '''
    a_by_types = dict()
    for i, a in enumerate(e['arguments']):
      if not a['entity_type'] in a_by_types:
        a_by_types[a['entity_type']] = [i]
      else:
        a_by_types[a['entity_type']].append(i)
    
    for a_type in a_by_types:
      for a in a_by_types[a_type]:
        a_token = a['head_lemma']
        a_prob = np.asarray([P_va[a_type][k][a_token] for k in range(self.Kv)])
        e_prob *= a_prob
    '''
    return e_prob 

  def event_event_prob(self, e1, e2, arg_modes):
    P_ee = self.event_coref_model.P_ee
    e1_token = e1['head_lemma']
    e2_token = e2['head_lemma']
    prob = P_ee[e1_token][e2_token]
    return prob
  
  def argument_argument_prob(self, e1, e2, arg_modes):
    P_ij = self.entity_coref_model.P_ij
    P_aa = self.entity_coref_model.P_ee
    a1_by_types = dict()
    for i, a1 in enumerate(e1['arguments']):
      if not a1['entity_type'] in a1_by_types:
        a1_by_types[a1['entity_type']] = [i]
      else:
        a1_by_types[a1['entity_type']].append(i)

    a2_by_types = dict()
    for i, (a2, arg_mode) in enumerate(zip(e2['arguments'], arg_modes)):
      if not a2['entity_type'] in a2_by_types:
        a2_by_types[a2['entity_type']] = [[i, arg_mode]]
      else:
        a2_by_types[a2['entity_type']].append([i, arg_mode])

    prob = 1.
    for a_type in a2_by_types:
      if not a_type in a1_by_types: 
        for a2_idx, arg_mode in a2_by_types[a_type]:
          e1_token = e1['head_lemma']
          a2 = e2['arguments'][a2_idx]
          a2_token = a2['head_lemma']
          d_sent = a2['sentence_id'] - e1['sentence_id']
          if not e1_token in self.P_ea[a_type]:
            prob *= P_ij[arg_mode][d_sent] * P_aa[arg_mode][NULL][a2_token]
          else:
            if not a2_token in self.P_ea[a_type][e1_token]:
              a2_token = UNK
            prob *= P_ij[arg_mode][d_sent] * self.P_ea[a_type][e1_token][a2_token]
      else:
        for a2_idx, arg_mode in a2_by_types[a_type]:
          a2 = e2['arguments'][a2_idx]
          a2_token = self.entity_coref_model.get_mode_feature(a2, arg_mode)
          a_prob = []
          for a1_idx in a1_by_types[a_type]:
            a1 = e1['arguments'][a1_idx]
            a1_token = self.entity_coref_model.get_mode_feature(a1, arg_mode) 
            d_sent = a2['sentence_id'] - a1['sentence_id']
            if self.entity_coref_model.is_match(a1, a2, arg_mode):
              a_prob.append(P_ij[arg_mode][d_sent] * P_aa[arg_mode][a1_token][a2_token])
            else:
              a_prob.append(P_ij[arg_mode][d_sent] * P_aa[arg_mode][NULL][a2_token])
            print(a1_token, a2_token, prob) # XXX 
          prob *= np.max(a_prob)
    return prob
            
  def predict_antecedents(self, event_features, action_features):
    antecedents = []
    cluster_ids = []
    scores_all = []
    event_modes_sents, argument_modes_sents = self.get_modes(event_features)

    n_cluster = 0
    for e_feat, v_feat, event_modes_sent, argument_modes_sent in zip(event_features, action_features, event_modes_sents, argument_modes_sents):
      v_prob = self.compute_cluster_prob(v_feat) 
      antecedent = [-1]*len(e_feat)
      cluster_id = [0]*len(e_feat)
      scores_all.append([])
      L = v_feat.shape[0]
      for e_idx, (e, event_mode, arg_modes) in enumerate(zip(e_feat, event_modes_sent, argument_modes_sent)):
        e_null = {'sentence_id': 0,
                  'tokens': NULL,
                  'head_lemma': NULL,
                  'arguments': []}

        scores = [self.event_event_prob(e_null, e, arg_modes)]
        for a_idx, a in enumerate(e_feat[:e_idx]):
          score = 0
          if self.is_match(e, a, event_mode):
            # If the current mention is in textual mode
            if event_mode == 'textual':
              score += self.event_event_prob(a, e, arg_modes)  
            # If the current mention is in visual mode
            elif event_mode == 'visual':
              e_prob = self.action_event_prob(e) 
              a_prob = self.action_event_prob(a)
              ve_prob = v_prob @ e_prob
              va_prob = v_prob @ a_prob
              ve_prob /= ve_prob.sum()
              va_prob /= va_prob.sum()
              score += ve_prob @ va_prob 
          scores.append(score)
        scores_all[-1].append(scores)
        scores = np.asarray(scores)

        antecedent[e_idx] = int(np.argmax(scores)) - 1
        
        # If antecedent idx == -1, the mention belongs to a new cluster; 
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
 
def load_event_features(config, action_labels, entity_label_dicts, split):
  lemmatizer = WordNetLemmatizer()
  event_feature_types = config['event_feature_types']
  event_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_events.json'), 'r', 'utf-8'))
  ontology_map = json.load(open(os.path.join(config['data_folder'], '../ontology_mapping.json')))
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}.json')))
  event_docs_embs = np.load(os.path.join(config['data_folder'], f'{split}_events_with_arguments_glove_embeddings.npz')) 
  doc_to_event_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in event_docs_embs}

  event_label_dicts = dict()
  event_feats = []
  doc_ids = []
  spans_all = []
  cluster_ids_all = []
  tokens_all = [] 

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
    action_label = action_labels[feat_idx]
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
        match_action_types = ontology_map[event['event_type']]
        # if len(set(match_action_types).intersection(set(action_label))) > 0:
        if len(match_action_types) > 0:
          event['is_visual'] = True
        else:
          event['is_visual'] = False
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

def load_entity_features(config, split):
  entity_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_entities.json'), 'r', 'utf-8'))
  doc = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}.json')))
  docs_embs = np.load(os.path.join(config['data_folder'], f'{split}_entities_glove_embeddings.npz')) 
  doc_to_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in docs_embs}

  label_dicts = {}
  entity_feats = []
  doc_ids = []
  spans_all = []
  cluster_ids_all = []
  tokens_all = [] 

  for m in entity_mentions:
    if not m['doc_id'] in label_dicts:
      label_dicts[m['doc_id']] = dict()
    span = (min(m['tokens_ids']), max(m['tokens_ids']))
    label_dicts[m['doc_id']][span] = deepcopy(m)      

  for feat_idx, doc_id in enumerate(sorted(label_dicts)): # XXX
    doc_embs = docs_embs[doc_to_feat[doc_id]]
    label_dict = label_dicts[doc_id]
    spans = sorted(label_dict)
    entities = []
    for span_idx, span in enumerate(spans):
      entity = deepcopy(label_dict[span])  
      entity['entity_embedding'] = doc_embs[span_idx, :300]
      entity['idx'] = span_idx 
      entity = get_number_and_mention_type(entity)
   
      label_dicts[doc_id][span] = deepcopy(entity)
      entities.append(entity)  
    cluster_ids = [label_dict[span]['cluster_id'] for span in spans]
    
    entity_feats.append(entities)
    doc_ids.append(doc_id)
    spans_all.append(spans)
    cluster_ids_all.append(np.asarray(cluster_ids))
    tokens_all.append([t[2] for t in doc[doc_id]])
  return entity_feats, label_dicts

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
  action_npz = np.load(os.path.join(config['data_folder'], f'{split}_mmaction_event_finetuned_crossmedia.npz')) # XXX f'{split}_events_event_type_labels.npz'
  action_label_npz = np.load(os.path.join(config['data_folder'], f'{split}_mmaction_event_feat_labels_average.npz'))

  doc_to_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in sorted(action_npz, key=lambda x:int(x.split('_')[-1]))}
  doc_to_label = {'_'.join(label_id.split('_')[:-1]):label_id for label_id in sorted(action_label_npz, key=lambda x:int(x.split('_')[-1]))}
  action_feats = [action_npz[doc_to_feat[doc_id]] for doc_id in sorted(label_dicts)] 
  action_labels_onehot = [action_label_npz[doc_to_label[doc_id]] for doc_id in sorted(label_dicts)]
  action_labels = [[action_classes[y] for y in np.argmax(action_label, axis=-1)] for action_label in action_labels_onehot]
  return action_feats, action_labels

def load_data(config):
  event_feats_train = []
  entity_feats_train = []
  action_feats_train = []
  action_labels_train = []
  doc_ids_train = []
  spans_train = []
  cluster_ids_train = []
  tokens_train = []
  for split in config['splits']['train']:
    cur_action_feats_train, cur_action_labels_train = load_visual_features(config, split=split)
    cur_entity_feats_train, entity_label_dicts = load_entity_features(config, split=split) 
    cur_event_feats_train,\
    cur_doc_ids_train,\
    cur_spans_train,\
    cur_cluster_ids_train,\
    cur_tokens_train = load_event_features(config, cur_action_labels_train, entity_label_dicts, split=split)
    
    action_feats_train.extend(cur_action_feats_train)
    action_labels_train.extend(cur_action_labels_train)
    event_feats_train.extend(cur_event_feats_train)
    entity_feats_train.extend(cur_entity_feats_train)
    doc_ids_train.extend(cur_doc_ids_train)
    spans_train.extend(cur_spans_train)
    cluster_ids_train.extend(cur_cluster_ids_train)
    tokens_train.extend(cur_tokens_train)    
  print(f'Number of training examples: {len(event_feats_train)}')
  
  event_feats_test = []
  entity_feats_test = []
  action_feats_test = []
  action_labels_test = []
  doc_ids_test = []
  spans_test = []
  cluster_ids_test = []
  tokens_test = []
  for split in config['splits']['test']:
    cur_action_feats_test, cur_action_labels_test = load_visual_features(config, split='test')
    cur_entity_feats_test, entity_label_dicts = load_entity_features(config, split=split)
    cur_event_feats_test,\
    cur_doc_ids_test,\
    cur_spans_test,\
    cur_cluster_ids_test,\
    cur_tokens_test = load_event_features(config, cur_action_labels_test, entity_label_dicts, split='test')
 
    action_feats_test.extend(cur_action_feats_test)
    action_labels_test.extend(cur_action_labels_test)
    event_feats_test.extend(cur_event_feats_test)
    entity_feats_test.extend(cur_entity_feats_test)
    doc_ids_test.extend(cur_doc_ids_test)
    spans_test.extend(cur_spans_test)
    cluster_ids_test.extend(cur_cluster_ids_test)
    tokens_test.extend(cur_tokens_test)
  print(f'Number of test examples: {len(event_feats_test)}')
  
  return event_feats_train,\
         entity_feats_train,\
         action_feats_train,\
         action_labels_train,\
         doc_ids_train,\
         spans_train,\
         cluster_ids_train,\
         tokens_train,\
         event_feats_test,\
         entity_feats_test,\
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
  entity_feats_train,\
  action_feats_train,\
  action_labels_train,\
  doc_ids_train,\
  spans_train,\
  cluster_ids_train,\
  tokens_train,\
  event_feats_test,\
  entity_feats_test,\
  action_feats_test,\
  action_labels_test,\
  doc_ids_test,\
  spans_test,\
  cluster_ids_test,\
  tokens_test = load_data(config)

  ## Model training
  aligner = AmodalSMTJointCoreferencer(event_feats_train+event_feats_test, entity_feats_train+entity_feats_test, action_feats_train+action_feats_test, config)
  aligner.train(3)
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
  logging.info('MUC - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'Bcubed - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CEAFe - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CoNLL - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(*conll_metrics))
