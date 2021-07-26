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
# Part of the code modified from vpyp: https://github.com/vchahun/vpyp/blob/master/vpyp/pyp.py

# logger = logging.getLogger(__name__)
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

class LocalRestaurant:
  """
    tables: a list [count_1, ..., count_T], 
            where count_t is the number of customers with at table t;
    name2table: a dictionary {x:t}, mapping name x to table t
    table2menu: a dictionary {t:k}, mapping from table t to menu (dish) k
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
    self.table2menu = {}
    self.table_names = []
    self.alpha0 = alpha0
    self.K_max = K
  
  def seat_to(self, k, c):
    self.ncustomers += 1
    tables = self.tables # shallow copy the tables to a local variable
    if not k in self.name2table:
      tables.append(1)
      self.name2table[k] = self.ntables
      self.table_names.append(k)
      self.table2menu[k] = c 
      self.ntables += 1
    else:
      i = self.name2table[k]
      tables[i] += 1
    if self.ntables > self.K_max:
      print(f'Warning in LocalRestaurant: number of table exceeds max limit, {self.ntables} > {self.K_max}') 

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
      del self.table2menu[k]
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

class HDPEventAligner(object):
  def __init__(self,
               event_features_train,
               alpha0, beta0,
               Kc, Kf):
    """
    Attributes:
        event_crp: a Restaurant object storing the distribution for the event clusters,
        feature_crps: a dict from feature type (str) to a list of Restaurant objects
        e_feats_train: a list of list of strs,
        cluster_ids: a list of list of ints
    """ 
    feature_types = sorted(Kf)
    self.e_feats_train = event_features_train
    # self.a_feats_train = entity_features_train
    self.alpha0 = alpha0
    self.beta0 = beta0
    self.Kc = Kc # Max number of event clusters 
    self.Kf = Kf # Size of event features vocabs
    self.event_crp = Restaurant(alpha0, Kc) # Keep track of counts for each event
    self.feature_crps = {feat_type:[] for feat_type in feature_types} # Keep track of counts for each feature within each event
    self.local_crps = [LocalRestaurant(beta0, len(e_feat)) for e_feat in self.e_feats_train] # Keep track of local counts for each event
    self.cluster_ids = [[] for _ in self.e_feats_train]
    self.table_ids = [[] for _ in self.e_feats_train]

  def prob(self, c, e):
    p = 1.
    if c in self.event_crp.name2table:
      table_idx = self.event_crp.name2table[c]
      for feat_type in e:
        if feat_type == 'arguments':
          for a in e['arguments']:
            p *= self.feature_crps['arguments'][table_idx].prob(a)
        else:
          # XXX if e[feat_type] != NULL:
          p *= self.feature_crps[feat_type][table_idx].prob(e[feat_type])
    else: 
      for feat_type in e:
        if feat_type == 'arguments':
          for a in e['arguments']:
            p *= Restaurant(self.alpha0, self.Kf['arguments']).prob(a)
        else:
          # XXX if e[feat_type] != NULL:
          p *= Restaurant(self.alpha0, self.Kf[feat_type]).prob(e[feat_type])
    return p
  
  def log_likelihood(self):
    ll = 0
    for crp in self.local_crps:
      ll += crp.log_likelihood()
 
    for feat_type in self.feature_crps:
      for crp in self.feature_crps[feat_type]:
        ll += crp.log_likelihood()
    return ll

  def gibbs_sample(self, e, crp, temp=-1): 
    """ Sample from P(z_ji|z^-ji, e_ji=e, a_ji=a, e^-ji, a^-ji) """
    # First, sample from P(t_ji|t_j^-ji, z^-ji, e_ji=e, a_ji=a, e_j^-ji, a_j^-ji)
    P = [crp.prob(-1) * self.prob(-1, e)]*self.Kc
    table_names = [-1]*self.Kc
    for t_idx, t in enumerate(crp.table_names):
      c = crp.table2menu[t]
      c_idx = self.event_crp.name2table[c]
      P[c_idx] = crp.prob(t) * self.prob(c, e)
      table_names[c_idx] = t

    if temp > 0:
      P = np.log(np.asarray(P)+EPS) / temp
      P = np.exp(P).tolist()
    P = np.asarray(P) / sum(P)
    
    table_probs = np.asarray([crp.prob(t) * self.prob(crp.table2menu[t], e) for t in crp.table_names])
    # print('table prior: ', [crp.prob(t) for t in crp.table_names])
    # print('feat likelhood: ', [self.prob(crp.table2menu[t], e) for t in crp.table_names])
    # print('table new prior: ', crp.prob(-1))
    # print('feat new likelihood: ', self.prob(-1, e))
    # print('\n')
  
    norm = sum(P)
    x = norm * random.random()
    for c, t, w in zip(self.event_crp.table_names, table_names, P):
      if x < w: return c, t
      x -= w      
    return -1, -1

  def train(self, n_iter=35, 
            out_dir='./'):
    # inv_temp_start, inv_temp_end = 0.1, 1
    # anneal_temps = 1. / np.linspace(inv_temp_start, inv_temp_end, n_iter)
    order = list(range(len(self.e_feats_train)))
    for i_iter in range(n_iter):
      random.shuffle(order)
      for i in order:
        e_feat = self.e_feats_train[i]
        new_cluster_ids = [-1]*len(e_feat)
        new_table_ids = [-1]*len(e_feat)

        mention_order = list(range(len(e_feat)))
        random.shuffle(mention_order) 
        for i_m in mention_order:
          if i_iter > 0: 
            c = self.cluster_ids[i][i_m]
            t = self.table_ids[i][i_m]
            e = e_feat[i_m]
            self.local_crps[i].unseat_from(t)
            self.unseat_from(c, e)
          
          for feat_type in self.feature_crps:
            assert len(self.feature_crps[feat_type]) == self.event_crp.ntables 
        
          e = e_feat[i_m] 
          c, t = self.gibbs_sample(e, self.local_crps[i]) # temp=anneal_temps[i_iter])
          if c == -1:
            new_c = 0
            while new_c in self.event_crp.name2table: # Create a new key for the new event
              new_c += 1
            c = new_c
          if t == -1:
            new_t = 0
            while new_t in self.local_crps[i].name2table:
              new_t += 1
            t = new_t

          self.local_crps[i].seat_to(t, c)
          self.seat_to(c, e)
          
          new_cluster_ids[i_m] = c
          new_table_ids[i_m] = t
        self.table_ids[i] = deepcopy(new_table_ids)
        self.cluster_ids[i] = deepcopy(new_cluster_ids)
          
      if i_iter % 10 == 0:
        self.save(out_dir)
      print(f'Iteration {i_iter}, log likelihood = {self.log_likelihood():.1f}')

  def cluster(self, 
              event_features_test):
    cluster_ids = []
    crp = LocalRestaurant(self.alpha0, self.Kd)
    for e_sent in zip(event_features_test): 
      cluster_ids.append([self.gibbs_sample(e, crp)[0] for e in e_sent])
    return cluster_ids

  def seat_to(self, c, e):
    if not c in self.event_crp.name2table: # Create CRPs for a new event
      self.event_crp.seat_to(c)
      for feat_type in e:
        new_feat_crp = Restaurant(self.alpha0, self.Kf[feat_type])
        if feat_type == 'arguments':
          for a in e['arguments']:
            new_feat_crp.seat_to(a)
        else:
          new_feat_crp.seat_to(e[feat_type])
        self.feature_crps[feat_type].append(new_feat_crp)
    else:
      self.event_crp.seat_to(c)
      table_idx = self.event_crp.name2table[c]
      for feat_type in e:
        if feat_type == 'arguments':
          for a in e['arguments']:
            self.feature_crps[feat_type][table_idx].seat_to(a)
        else:
          self.feature_crps[feat_type][table_idx].seat_to(e[feat_type])
  
  def unseat_from(self, c, e):
    table_idx = self.event_crp.name2table[c] 
    self.event_crp.unseat_from(c)

    for feat_type in e:
      if feat_type == 'arguments':
        for a in e['arguments']:
          self.feature_crps['arguments'][table_idx].unseat_from(a)
      else:
        self.feature_crps[feat_type][table_idx].unseat_from(e[feat_type])

      if self.feature_crps[feat_type][table_idx].ntables == 0: # Replace the empty restaurant with the last restaurant
        self.feature_crps[feat_type][table_idx] = deepcopy(self.feature_crps[feat_type][-1])
        del self.feature_crps[feat_type][-1]
        assert self.event_crp.ntables == len(self.feature_crps[feat_type])
      
  def save(self, out_dir='./'):
    self.event_crp.save(os.path.join(out_dir, 'event_crp_'))
    out_str = ''
    f_out = open(os.path.join(out_dir, 'feature_crps_tables.txt'), 'w')
    for c in self.event_crp.table_names:
      table_idx = self.event_crp.name2table[c]
      for feat_type in self.feature_crps:
        out_str += f'{feat_type}\n'
        out_str += self.feature_crps[feat_type][table_idx].save(returnStr=True)
      out_str += '\n'
    f_out.write(out_str)
    f_out.close()

    f_out = open(os.path.join(out_dir, 'local_crps_tables.txt'), 'w')
    out_str = ''
    for doc_idx, crp in enumerate(self.local_crps):
      out_str += f'{doc_idx}\n'
      out_str += crp.save(returnStr=True)
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
  feature_types = config['feature_types']
  event_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_events.json'), 'r', 'utf-8'))
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}.json')))

  label_dicts = {}
  event_feats = []
  doc_ids = []
  spans_all = []
  cluster_ids_all = []
  tokens_all = [] 

  for m in event_mentions:
      if not m['doc_id'] in label_dicts:
        label_dicts[m['doc_id']] = {}
      token = lemmatizer.lemmatize(m['tokens'].lower(), pos='v')
      span = (min(m['tokens_ids']), max(m['tokens_ids']))
      label_dicts[m['doc_id']][span] = {'token_id': token,
                                        'cluster_id': m['cluster_id']}

      for feat_type in feature_types:
        if feat_type == 'arguments':
          label_dicts[m['doc_id']][span]['arguments'] = [a['head_lemma'] for a in m['arguments']]
          if len(m['arguments']) == 0:
            label_dicts[m['doc_id']][span]['arguments'].append(NULL)
        else:
          label_dicts[m['doc_id']][span][feat_type] = m[feat_type] 
        
  for feat_idx, doc_id in enumerate(sorted(label_dicts)): # XXX
    label_dict = label_dicts[doc_id]
    spans = sorted(label_dict)
    events = [{feat_type: label_dict[span][feat_type] for feat_type in feature_types} for span in spans] 
    cluster_ids = [label_dict[span]['cluster_id'] for span in spans]
    
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

def load_data(config):
  lemmatizer = WordNetLemmatizer() 
  event_mentions_train = json.load(codecs.open(os.path.join(config['data_folder'], 'train_events.json'), 'r', 'utf-8'))
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], 'train.json')))
  event_mentions_test = json.load(codecs.open(os.path.join(config['data_folder'], 'test_events.json'), 'r', 'utf-8'))
  doc_test = json.load(codecs.open(os.path.join(config['data_folder'], 'test.json')))
  feature_types = config['feature_types']

  vocab_feats = {feat_type:dict() for feat_type in feature_types}
  vocab_feats_freq = {feat_type:dict() for feat_type in feature_types}
  for m in event_mentions_train + event_mentions_test:    
    for feat_type in feature_types: 
      if feat_type == 'arguments':
        for a in m['arguments']:
          if not a['head_lemma'] in vocab_feats['arguments']:
            vocab_feats['arguments'][a['head_lemma']] = len(vocab_feats['arguments'])
            vocab_feats_freq['arguments'][a['head_lemma']] = 1
          else:
            vocab_feats_freq['arguments'][a['head_lemma']] += 1
      else:
        if not m[feat_type] in vocab_feats[feat_type]:
          vocab_feats[feat_type][m[feat_type]] = len(vocab_feats[feat_type])
          vocab_feats_freq[feat_type][m[feat_type]] = 1
        else:
          vocab_feats_freq[feat_type][m[feat_type]] += 1

  json.dump(vocab_feats_freq, open('vocab_feats_freq.json', 'w'), indent=2)

  event_feats_train,\
  doc_ids_train,\
  spans_train,\
  cluster_ids_train,\
  tokens_train = load_text_features(config, split='train')
  print(f'Number of training examples: {len(event_feats_train)}')
  
  event_feats_test,\
  doc_ids_test,\
  spans_test,\
  cluster_ids_test,\
  tokens_test = load_text_features(config, split='test')
  print(f'Number of test examples: {len(event_feats_test)}')
  
  return event_feats_train,\
         doc_ids_train,\
         spans_train,\
         cluster_ids_train,\
         tokens_train,\
         event_feats_test,\
         doc_ids_test,\
         spans_test,\
         cluster_ids_test,\
         tokens_test,\
         vocab_feats
 
if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', '-c', default='../configs/config_hdp_video_m2e2.json')
  parser.add_argument('--compute_confidence_bound', action='store_true')
  args = parser.parse_args()

  config_file = args.config
  config = pyhocon.ConfigFactory.parse_file(config_file)
  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  logging.basicConfig(filename=os.path.join(config['model_path'], 'train.log'), level=logging.DEBUG)

  event_feats_train,\
  doc_ids_train,\
  spans_train,\
  cluster_ids_train,\
  tokens_train,\
  event_feats_test,\
  doc_ids_test,\
  spans_test,\
  cluster_ids_test,\
  tokens_test,\
  vocab_feats = load_data(config)
  Kc = 2000 # max(max(cluster_ids) for cluster_ids in cluster_ids_train+cluster_ids_test)
  Kf = {feat_type:len(vocab_feats[feat_type]) for feat_type in vocab_feats}
  print(f'Max num of events: {Kc}')
  for feat_type in Kf:
    print(f'{feat_type} vocab size: {Kf[feat_type]}')

  pairwises = []
  mucs = []
  b_cubeds = []
  ceafes = []
  avgs = []
  
  ## Model training
  aligner = HDPEventAligner(event_feats_train+event_feats_test, 
                            alpha0=config['alpha0'],
                            beta0=config['beta0'],
                            Kc=Kc,
                            Kf=Kf)
  aligner.train(100, out_dir=config['model_path'])

  ## Test and evaluation
  pred_cluster_ids = [np.asarray(cluster_ids)+1 for cluster_ids in aligner.cluster_ids[len(event_feats_train):]]
  pred_labels = [torch.LongTensor(to_pairwise(a)) for a in pred_cluster_ids if a.shape[0] > 1]
  gold_labels = [torch.LongTensor(to_pairwise(c)) for c in cluster_ids_test if c.shape[0] > 1]
  pred_labels = torch.cat(pred_labels)
  gold_labels = torch.cat(gold_labels)

  # Compute pairwise scores
  pairwise_eval = Evaluation(pred_labels, gold_labels)  
  pairwise = [pairwise_eval.get_precision(), pairwise_eval.get_recall(), pairwise_eval.get_f1()]
  print(f'Pairwise - Precision: {pairwise_eval.get_precision()}, Recall: {pairwise_eval.get_recall()}, F1: {pairwise_eval.get_f1()}')
  logging.info(f'Pairwise precision: {pairwise_eval.get_precision()}, recall: {pairwise_eval.get_recall()}, F1: {pairwise_eval.get_f1()}')
  
  # Compute CoNLL scores and save readable predictions
  conll_eval = CoNLLEvaluation()

  for seed in config.seeds:
    f_out = open(os.path.join(config['model_path'], f'prediction_{seed}.readable'), 'w')
    random.seed(seed)
    np.random.seed(seed)
    logging.info(f'random seed = {seed}')

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
    pairwises.append(pairwise)
    mucs.append(muc)
    b_cubeds.append(b_cubed)
    ceafes.append(ceafe)
    avgs.append(avg)

    logging.info('MUC - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
                'Bcubed - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
                'CEAFe - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
                'CoNLL - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(*conll_metrics))
  mean_pairwise, std_pairwise = np.mean(np.asarray(pairwises), axis=0), np.std(np.asarray(pairwises), axis=0)
  mean_muc, std_muc = np.mean(np.asarray(mucs), axis=0), np.std(np.asarray(mucs), axis=0)
  mean_bcubed, std_bcubed = np.mean(np.asarray(b_cubeds), axis=0), np.std(np.asarray(b_cubeds), axis=0)
  mean_ceafe, std_ceafe = np.mean(np.asarray(ceafes), axis=0), np.std(np.asarray(ceafes), axis=0)
  mean_avg, std_avg = np.mean(np.asarray(avgs), axis=0), np.std(np.asarray(avgs), axis=0)
  print(f'Pairwise: precision {mean_pairwise[0]} +/- {std_pairwise[0]}, '
        f'recall {mean_pairwise[1]} +/- {std_pairwise[1]}, '
        f'f1 {mean_pairwise[2]} +/- {std_pairwise[2]}')
  print(f'MUC: precision {mean_muc[0]} +/- {std_muc[0]}, '
        f'recall {mean_muc[1]} +/- {std_muc[1]}, '
        f'f1 {mean_muc[2]} +/- {std_muc[2]}')
  print(f'Bcubed: precision {mean_bcubed[0]} +/- {std_bcubed[0]}, '
        f'recall {mean_bcubed[1]} +/- {std_bcubed[1]}, '
        f'f1 {mean_bcubed[2]} +/- {std_bcubed[2]}')
  print(f'CEAFe: precision {mean_ceafe[0]} +/- {std_ceafe[0]}, '
        f'recall {mean_ceafe[1]} +/- {std_ceafe[1]}, '
        f'f1 {mean_ceafe[2]} +/- {std_ceafe[2]}')
  print(f'CoNLL: precision {mean_avg[0]} +/- {std_avg[0]}, '
        f'recall {mean_avg[1]} +/- {std_avg[1]}, '
        f'f1 {mean_avg[2]} +/- {std_avg[2]}')