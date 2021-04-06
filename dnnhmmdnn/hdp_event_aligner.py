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
import nltk
from nltk.stem import WordNetLemmatizer
from evaluator import Evaluation, CoNLLEvaluation
random.seed(2)
# Part of the code modified from vpyp: https://github.com/vchahun/vpyp/blob/master/vpyp/pyp.py

logger = logging.getLogger(__name__)
class Restaurant:
  # Attributes:
  # ----------
  #   tables: a list [count_1, ..., count_T], 
  #           where count_t is the number of customers with at table t;
  #   name2table: a dictionary {k:t}, mapping name k to table t
  #   ncustomers: sum(tables),
  #               storing the total number of customers with each dish; 
  #   ntables: len(tables),
  #            total number of tables;
  #   p_init: a dictionary {k: p_0(k)},
  #         where p_0(k) is the initial probability for table with name k
  #   alpha0: concentration, Dirichlet process parameter
  def __init__(self, alpha0):
    self.tables = []
    self.ntables = 0
    self.ncustomers = 0
    self.name2table = {}
    self.table_names = []
    self.p_init = {}
    self.alpha0 = alpha0

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
    w = 0. 
    if k in self.name2table:
      i = self.name2table[k]
      w += self.tables[i]
    else:
      w = self.alpha0
    
    return w / (self.alpha0 + self.ncustomers) 

  def log_likelihood(self):
    ll = math.lgamma(self.alpha0) - math.lgamma(self.alpha0 + self.ncustomers)
    ll += sum(math.lgamma(self.tables[i]) for i, k in enumerate(self.table_names))
    return ll

  def save(self, outputDir='./', returnStr=False):
    sorted_indices = sorted(list(range(self.ntables)), key=lambda x:self.tables[x], reverse=True)
    outStr = ''
    for i in sorted_indices[:10]:
      outStr += '%s %d\n' % (self.table_names[i], self.tables[i])

    if returnStr:
      return outStr
    else:
      with open(outputDir + 'tables.txt', 'w') as f:
        f.write(outStr)

class HDPEventAligner(object):
  def __init__(self,
               event_features_train,
               entity_features_train,
               alpha0):
    """
    Attributes:
        event_crp: a Restaurant object storing the distribution for the event clusters,
        argument_crp: a Restaurant object storing the distribution for the entity/argument clusters,
        e_feats_train: a list of list of strs,
        a_feats_train: a list of list of list of strs,
        cluster_ids: a list of list of ints
    """ 
    self.e_feats_train = event_features_train
    self.a_feats_train = entity_features_train
    self.alpha0 = alpha0
    self.event_crp = Restaurant(alpha0) # Keep track of counts for each event
    self.feature_crps = [] # Keep track of counts for each feature within each event
    self.cluster_ids = [[] for _ in self.e_feats_train]

  def prob(self, c, e, a):
    p = self.event_crp.prob(c)
    if c in self.event_crp.name2table:
      table_idx = self.event_crp.name2table[c]
      p *= self.feature_crps[table_idx][0].prob(e)
      p *= np.prod([self.feature_crps[table_idx][1].prob(a_i) for a_i in a])
    return p
  
  def log_likelihood(self):
    ll = self.event_crp.log_likelihood()
    for table_idx in range(self.event_crp.ntables):
      for i in range(2):
        ll += self.feature_crps[table_idx][i].log_likelihood()
    return ll

  def gibbs_sample(self, e, a):
    P = [self.prob(c, e, a) for c in self.event_crp.table_names]
    P.append(self.event_crp.prob(-1))
    norm = sum(P)
    x = norm * random.random()
    for c, w in zip(self.event_crp.table_names+[-1], P):
      if x < w: return c
      x -= w
    return -1

  def train(self, n_iter=35, out_dir='./'):
    order = list(range(len(self.e_feats_train)))
    for i_iter in range(n_iter):
      for i in order:
        e_feat = self.e_feats_train[i]
        a_feat = self.a_feats_train[i]
        if i_iter > 0:
          for c, e, a in zip(self.cluster_ids[i], e_feat, a_feat):
            self.unseat_from(c, e, a)
        assert len(self.feature_crps) == self.event_crp.ntables 

        new_cluster_ids = []
        for e, a in zip(e_feat, a_feat):
          c = self.gibbs_sample(e, a)
          if c == -1:
            new_c = 0
            while new_c in self.event_crp.name2table: # Create a new key for the new event
              new_c += 1
            c = new_c
          self.seat_to(c, e, a)
          new_cluster_ids.append(c)
        self.cluster_ids[i] = deepcopy(new_cluster_ids)
          
      if i_iter % 10 == 0:
        self.save(out_dir)
      print(f'Iteration {i_iter}, log likelihood = {self.log_likelihood():.1f}')

  def cluster(self, 
              event_features_test,
              entity_features_test):
    cluster_ids = []
    for e_feat, a_feat in zip(event_features_test, entity_features_test):
      table_idxs = [np.argmax([self.prob(c, e, a) for c in self.event_crp.table_names])\
                                                            for e, a in zip(e_feat, a_feat)]
      cluster_ids.append(np.asarray([self.event_crp.table_names[table_idx] for table_idx in table_idxs]))
    return cluster_ids

  def seat_to(self, c, e, a):
    if not c in self.event_crp.name2table: # Create CRPs for a new event
      self.event_crp.seat_to(c)
      new_trigger_crp = Restaurant(self.alpha0)
      new_argument_crp = Restaurant(self.alpha0)
      new_trigger_crp.seat_to(e)
      for a_i in a:
        new_argument_crp.seat_to(a_i)
      self.feature_crps.append([new_trigger_crp, new_argument_crp])
    else:
      self.event_crp.seat_to(c)
      table_idx = self.event_crp.name2table[c]
      self.feature_crps[table_idx][0].seat_to(e)
      for a_i in a:
        self.feature_crps[table_idx][1].seat_to(a_i)
  
  def unseat_from(self, c, e, a):
    table_idx = self.event_crp.name2table[c] 
    self.event_crp.unseat_from(c)
    self.feature_crps[table_idx][0].unseat_from(e)
    for a_i in a:
      self.feature_crps[table_idx][1].unseat_from(a_i) 

    if self.feature_crps[table_idx][0].ntables == 0: # Replace the empty restaurant with the last restaurant
      self.feature_crps[table_idx] = deepcopy(self.feature_crps[-1])
      del self.feature_crps[-1]
      assert self.event_crp.ntables == len(self.feature_crps)
      
  def save(self, out_dir='./'):
    self.event_crp.save(os.path.join(out_dir, 'event_crp_'))
    feat_names = ['triggers', 'arguments']
    out_str = ''
    f_out = open(os.path.join(out_dir, 'feature_crps_tables.txt'), 'w')
    for c in self.event_crp.table_names:
      table_idx = self.event_crp.name2table[c]
      for i in range(2):
        out_str += f'{feat_names[i]}\n'
        out_str += self.feature_crps[table_idx][i].save(returnStr=True)
        out_str += '\n'
    f_out.write(out_str)
    f_out.close()

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
    
def load_text_features(config, split):
  lemmatizer = WordNetLemmatizer() 
  event_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_events.json'), 'r', 'utf-8'))
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}.json')))

  label_dicts = {}
  event_feats = []
  entity_feats = []
  doc_ids = []
  spans_all = []
  spans_entity_all = []
  cluster_ids_all = []
  tokens_all = [] 

  for m in event_mentions:
      if not m['doc_id'] in label_dicts:
        label_dicts[m['doc_id']] = {}
      token = lemmatizer.lemmatize(m['tokens'].lower(), pos='v')
      span = (min(m['tokens_ids']), max(m['tokens_ids']))
      label_dicts[m['doc_id']][span] = {'token_id': token,
                                        'cluster_id': m['cluster_id'],
                                        'arguments': {}} 
      
      for a in m['arguments']:
        a_token = lemmatizer.lemmatize(a['text'].lower())
        label_dicts[m['doc_id']][span]['arguments'][(a['start'], a['end'])] = a_token

  for feat_idx, doc_id in enumerate(sorted(label_dicts)): # XXX
    label_dict = label_dicts[doc_id]
    spans = sorted(label_dict)
    a_spans = [[a_span for a_span in sorted(label_dict[span]['arguments'])] for span in spans]
    events = [label_dict[span]['token_id'] for span in spans]
    entities = [[label_dict[span]['arguments'][a_span] for a_span in sorted(label_dict[span]['arguments'])] for span in spans]
    cluster_ids = [label_dict[span]['cluster_id'] for span in spans]
    
    event_feats.append(events)
    entity_feats.append(entities)
    doc_ids.append(doc_id)
    spans_all.append(spans)
    spans_entity_all.append(a_spans)    
    cluster_ids_all.append(np.asarray(cluster_ids))
    tokens_all.append([t[2] for t in doc_train[doc_id]])
  return event_feats,\
    entity_feats,\
    doc_ids,\
    spans_all,\
    spans_entity_all,\
    cluster_ids_all,\
    tokens_all

def load_data(config):
  lemmatizer = WordNetLemmatizer() 
  event_mentions_train = json.load(codecs.open(os.path.join(config['data_folder'], 'train_events.json'), 'r', 'utf-8'))
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], 'train.json')))
  event_mentions_test = json.load(codecs.open(os.path.join(config['data_folder'], 'test_events.json'), 'r', 'utf-8'))
  doc_test = json.load(codecs.open(os.path.join(config['data_folder'], 'test.json')))

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
      argument = a['text']
      argument = lemmatizer.lemmatize(argument.lower())
      if not argument in vocab_entity:
        vocab_entity[argument] = len(vocab_entity)
        vocab_entity_freq[argument] = len(vocab_entity)
      else:
        vocab_entity_freq[argument] += 1

  json.dump(vocab_freq, open('vocab_freq.json', 'w'), indent=2)
  json.dump(vocab_entity_freq, open('vocab_entity_freq.json', 'w'), indent=2)

  vocab_size = len(vocab)
  vocab_entity_size = len(vocab_entity)
  print(f'Vocab size: {vocab_size}, vocab entity size: {vocab_entity_size}')

  event_feats_train,\
  entity_feats_train,\
  doc_ids_train,\
  spans_train,\
  spans_entity_train,\
  cluster_ids_train,\
  tokens_train = load_text_features(config, split='train')
  print(f'Number of training examples: {len(event_feats_train)}')
  
  event_feats_test,\
  entity_feats_test,\
  doc_ids_test,\
  spans_test,\
  spans_entity_test,\
  cluster_ids_test,\
  tokens_test = load_text_features(config, split='test')
  print(f'Number of test examples: {len(event_feats_test)}')
  
  return event_feats_train,\
         entity_feats_train,\
         doc_ids_train,\
         spans_train,\
         spans_entity_train,\
         cluster_ids_train,\
         tokens_train,\
         event_feats_test,\
         entity_feats_test,\
         doc_ids_test,\
         spans_test,\
         spans_entity_test,\
         cluster_ids_test,\
         tokens_test,\
         vocab,\
         vocab_entity
 
if __name__ == '__main__':
  config_file = '../configs/config_hdp_video_m2e2.json'
  config = pyhocon.ConfigFactory.parse_file(config_file)
  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  logging.basicConfig(filename=os.path.join(config['model_path'], 'train.log'))

  event_feats_train,\
  entity_feats_train,\
  doc_ids_train,\
  spans_train,\
  spans_entity_train,\
  cluster_ids_train,\
  tokens_train,\
  event_feats_test,\
  entity_feats_test,\
  doc_ids_test,\
  spans_test,\
  spans_entity_test,\
  cluster_ids_test,\
  tokens_test,\
  vocab, vocab_entity = load_data(config)
  Ke = len(vocab)
  Ka = len(vocab_entity)

  ## Model training
  aligner = HDPEventAligner(event_feats_train, 
                            entity_feats_train,
                            alpha0=1.)
  aligner.train(35, out_dir=config['model_path'])

  ## Test and evaluation
  conll_eval = CoNLLEvaluation()
  pred_cluster_ids = aligner.cluster(event_feats_test, entity_feats_test)
  pred_labels = [torch.LongTensor(to_pairwise(a)) for a in pred_cluster_ids if a.shape[0] > 1]
  gold_labels = [torch.LongTensor(to_pairwise(c)) for c in cluster_ids_test if c.shape[0] > 1]
  pred_labels = torch.cat(pred_labels)
  gold_labels = torch.cat(gold_labels)

  # Compute pairwise scores
  pairwise_eval = Evaluation(pred_labels, gold_labels)  
  print(f'Pairwise - Precision: {pairwise_eval.get_precision():.4f}, Recall: {pairwise_eval.get_recall():.4f}, F1: {pairwise_eval.get_f1():.4f}')
  logger.info(f'Pairwise precision: {pairwise_eval.get_precision()}, recall: {pairwise_eval.get_recall()}, F1: {pairwise_eval.get_f1()}')
  
  # Compute CoNLL scores and save readable predictions
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
  logger.info('MUC - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'Bcubed - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CEAFe - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CoNLL - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(*conll_metrics))
