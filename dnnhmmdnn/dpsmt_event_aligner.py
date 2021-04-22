import math
import random
import numpy as np
from copy import deepcopy
import numpy as np
import logging
import json
import os
import pyhocon
import itertools
import codecs
import torch
import argparse
import nltk
from nltk.stem import WordNetLemmatizer
from evaluator import Evaluation, CoNLLEvaluation
random.seed(2)
EPS = 1e-100
NULL = '###NULL###'

class Restaurant:
  """
  Attributes:
     tables: a list [count_1, ..., count_T], 
             where count_t is the number of customers with at table t;
     name2table: a dictionary {k:t}, mapping name k to table t
     ncustomers: sum(tables),
                 storing the total number of customers with each dish; 
     ntables: len(tables),
              total number of tables;
     alpha0: concentration, Dirichlet process parameter
     K_max: maximum number of tables 
  """
  def __init__(self, alpha0, K):
    self.tables = []
    self.ntables = 0
    self.ncustomers = 0
    self.name2table = {}
    self.table_names = []
    self.alpha0 = alpha0
    self.K_max = K

  def seat_to(self, k):
    self.ncustomers += 1 
    tables = self.tables # shallow copy the tables to a local variable
    if not k in self.name2table: # add a new table
      tables.append(1)
      self.name2table[k] = self.ntables
      self.table_names.append(k)
      self.ntables += 1
    else:
      i = self.name2table[k]
      tables[i] += 1
    if self.ntables > self.K_max:
      print('Warning: number of table exceeds max limit') 

  def unseat_from(self, k):
    self.ncustomers -= 1
    i = self.name2table[k]
    tables = self.tables
    tables[i] -= 1
    if tables[i] == 0: # cleanup empty table
      k_new = self.table_names[-1] 
      self.table_names[i] = k_new # replace the empty table with the last table
      self.name2table[k_new] = i
      self.tables[i] = self.tables[-1]
      del self.name2table[k] 
      del self.table_names[-1]
      del self.tables[-1]
      self.ntables -= 1 

  def prob(self, k):
    w = self.alpha0 / self.K_max 
    
    if k in self.name2table:
      i = self.name2table[k]
      w += self.tables[i] 
    return w / (self.alpha0 + self.ncustomers) 

  def log_likelihood(self):
    ll = math.lgamma(self.alpha0)\
         - self.ntables * math.lgamma(self.alpha0 / self.K_max)\
         + sum(math.lgamma(self.tables[i] + self.alpha0 / self.K_max) for i in range(self.ntables))\
         - math.lgamma(self.ncustomers + self.alpha0)         
    return ll

  def save(self, outputDir='./', returnStr=False):
    sorted_indices = sorted(list(range(self.ntables)), key=lambda x:self.tables[x], reverse=True)
    outStr = ''
    for i in sorted_indices[:10]:
      outStr += 'Table %d: %s %d\n' % (i, self.table_names[i], self.tables[i])

    if returnStr:
      return outStr
    else:
      with open(outputDir + 'tables.txt', 'w') as f:
        f.write(outStr)

class DirichletTranslationEventAligner(object):
  def __init__(self,
               action_features_train,
               event_features_train,
               alpha0, beta0,
               vocab, Kv):
    """
    Attributes:
      alignment_crps: a list of Restaurant objects storing the distribution for the event-action alignments
      translation_crps: a list of Restaurant objects storing the distribution of action classes for each trigger word
    """
    self.e_feats_train = event_features_train
    self.v_feats_train = action_features_train

    for i in range(len(action_feats_train)):
      vocab[NULL] = len(vocab)
      Nv = self.v_feats_train[i].shape[0]
      v_feat_new = np.zeros((Nv+1, Kv+1), dtype=np.int)
      v_feat_new[:Nv, :Kv] = self.v_feats_train[i].astype(int)
      v_feat_new[Nv, Kv] = 1
      self.v_feats_train[i] = deepcopy(v_feat_new)

    self.Ke = len(vocab)
    self.Kv = Kv + 1
    self.vocab = vocab
    self.alpha0 = alpha0
    self.beta0 = beta0

    self.alignment_crps = [Restaurant(alpha0, len(v_feat)) for v_feat in self.v_feats_train]
    self.translation_crps = [Restaurant(beta0, Kv) for _ in range(self.Ke)] 
    self.alignments = [[] for _ in self.e_feats_train]

  def prob(self, e, v):
    v_label = np.argmax(v) # XXX
    return self.translation_crps[self.vocab[e]].prob(v_label) 

  def log_likelihood(self):
    ll = 0
    for crp in self.alignment_crps:
      ll += crp.log_likelihood()

    for crp in self.translation_crps:
      ll += crp.log_likelihood()
    return ll 

  def gibbs_sample(self, e, vs, crp):
    """ Sample from P(a_ji|a^-ji, e_ji=e, v_ji=v, e^-ji) """
    prior = [crp.prob(j) for j in range(len(vs))]
    P = [crp.prob(j) * self.prob(e, v) for j, v in enumerate(vs)]

    norm = sum(P)
    x = norm * random.random()
    for j, w in enumerate(P):
      if x < w: return j
      x -= w
    return j    

  def train(self, n_iter,
            out_dir='./'):
    order = list(range(len(self.e_feats_train)))
    for i_iter in range(n_iter):
      random.shuffle(order)
      for i in order:
        e_feat = self.e_feats_train[i]
        v_feat = self.v_feats_train[i]

        new_alignment = [-1]*len(e_feat)
        mention_order = list(range(len(e_feat)))
        random.shuffle(mention_order) 
        for i_m in mention_order:
          e = e_feat[i_m]
          if i_iter > 0:
            a_i = self.alignments[i][i_m]
            v = np.argmax(v_feat[a_i]) # XXX
            if v < self.Kv - 1:
              self.alignment_crps[i].unseat_from(a_i)
              self.translation_crps[self.vocab[e]].unseat_from(v) 
          
          a_i = self.gibbs_sample(e, v_feat, self.alignment_crps[i])
          v = np.argmax(v_feat[a_i]) # XXX
          new_alignment[i_m] = a_i
          if v == self.Kv - 1: # Skip if v is NULL
            continue
          self.alignment_crps[i].seat_to(a_i)
          self.translation_crps[self.vocab[e]].seat_to(v)
        self.alignments[i] = deepcopy(new_alignment)
      
      if i_iter % 10 == 0:
        self.save(out_dir)
      print(f'Iteration {i_iter}, log likelihood = {self.log_likelihood():.1f}')

  def align(self, 
            action_features_test,
            event_features_test):
    alignments = []
    for e_feat, v_feat in zip(event_features_test, action_features_test):
      crp = Restaurant(self.alpha0, len(e_feat))
      alignment = []
      for e, v in zip(e_feat, v_feat):
        alignment.append(self.gibbs_sample(e, v, crp))
      alignments.append(alignment)
    return alignments

  def save(self, out_dir='./'):
    out_str = ''
    f_out = open(os.path.join(out_dir, 'alignment_crps.txt'), 'w')
    for doc_idx, crp in enumerate(self.alignment_crps):
      out_str += f'{doc_idx}\n'
      out_str += crp.save(returnStr=True)
      out_str += '\n'
    f_out.write(out_str)
    f_out.close()

    out_str = ''
    f_out = open(os.path.join(out_dir, 'translation_crps.txt'), 'w')
    vocabs = sorted(self.vocab, key=lambda x:self.vocab[x])
    for e_idx, crp in enumerate(self.translation_crps):
      out_str += f'{vocabs[e_idx]}\n'
      out_str += crp.save(returnStr=True)
      out_str += '\n'
    f_out.write(out_str)
    f_out.close()

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
  event_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_events_with_linguistic_features.json'), 'r', 'utf-8'))
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
      span = (min(m['tokens_ids']), max(m['tokens_ids']))
      label_dicts[m['doc_id']][span] = {'token_id': m['head_lemma'],
                                        'cluster_id': m['cluster_id'],
                                        'type': m['event_type'],
                                        'arguments': {}} 
      
      for a in m['arguments']:
        if 'text' in a:
          a_token = a['head_lemma']
          a_span = (a['start'], a['end'])
        else:
          a_token = a['head_lemma']
          a_span = (min(a['tokens_ids']), max(a['tokens_ids']))
        label_dicts[m['doc_id']][span]['arguments'][a_span] = {'token_id': a_token,
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

    event_feats.append(events)
    entity_feats.append(entities)
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
  event_mentions_train = json.load(codecs.open(os.path.join(config['data_folder'], 'train_events_with_linguistic_features.json'), 'r', 'utf-8'))
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], 'train.json')))
  event_mentions_test = json.load(codecs.open(os.path.join(config['data_folder'], 'test_events_with_linguistic_features.json'), 'r', 'utf-8'))
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
    trigger = m['head_lemma']
    if not trigger in vocab:
      vocab[trigger] = len(vocab)
      vocab_freq[trigger] = 1
    else:
      vocab_freq[trigger] += 1
    
    for a in m['arguments']:
      argument = a['head_lemma']
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
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', '-c', default='../configs/config_dpsmt_video_m2e2.json')
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
  Kv = len(action_classes)

  ## Model training 
  aligner = DirichletTranslationEventAligner(action_feats_train+action_feats_test,
                                             event_feats_train+event_feats_test,
                                             alpha0=config['alpha0'], 
                                             beta0=config['beta0'],
                                             vocab=vocab,
                                             Kv=Kv)
  aligner.train(100, out_dir=config['model_path'])

  ## Test and evaluation
  pred_cluster_ids = [np.asarray(cluster_ids) for cluster_ids in aligner.alignments[len(event_feats_train):]]
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
  for doc_id, token, span, pred_cluster_id, gold_cluster_id in zip(doc_ids_test, tokens_test, spans_test, pred_cluster_ids, cluster_ids_test):
    antecedent = to_antecedents(pred_cluster_id)
    pred_clusters, gold_clusters = conll_eval(torch.LongTensor(span),
                                              torch.LongTensor(antecedent),
                                              torch.LongTensor(span),
                                              torch.LongTensor(gold_cluster_id)) 
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
