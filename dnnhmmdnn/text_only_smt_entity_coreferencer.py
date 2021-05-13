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
from evaluator import Evaluation, CoNLLEvaluation
np.random.seed(2)
random.seed(2)

NULL = '###NEW###'
PLURAL_PRON = ['they', 'them', 'those', 'these', 'we', 'us', 'their', 'our']
SINGULAR = 'singular'
PLURAL = 'plural'
PROPER = 'proper'
NOMINAL = 'nominal'
PRON = 'pronoun'
MODE_S = 'semantic'
MODE_D = 'discourse'

EPS = 1e-100
class SMTEntityCoreferencer:
  def __init__(self, entity_features, config):
    '''
    Unsupervised coreference system based on
    ``Unsupervised Ranking Model for Entity Coreference Resolution``, 
    X. Ma, Z. Liu and E. Hovy, NAACL, 2016.
    
    :param doc_path: path to the mapping of
            [doc_id]: list of [sent id, token id, token, is entity/entity]
    :param mention_path: path to the list of dicts of:
            {'doc_id': str,
             'sentence_id': str,
             'tokens_ids': list of ints,
             'cluster_id': '0',
             'tokens': str, tokens concatenated with space} 
    '''
    self.config = config
    self.e_feats_train = entity_features
    self.ee_counts = dict()
    self.modes = [MODE_S, MODE_D]
    self.P_ee = {mode: dict() for mode in self.modes}
    self.P_ij = {mode: dict() for mode in self.modes}
    self.vocab = self.get_vocab(entity_features) 
    self.modes_sents = self.get_modes(entity_features)

    self.initialize()
    logging.info(f'Number of documents = {len(self.e_feats_train)}, vocab size = {len(self.vocab)}')

  def get_modes(self, entity_feats):
    modes_sents = []
    for e_feat in entity_feats:
      modes_sent = []
      for e_idx, e in enumerate(e_feat):
        mode = MODE_D
        for a_idx, a in enumerate(e_feat[:e_idx]):
          if self.is_match(e, a, mode=MODE_S):
            mode = MODE_S
            break
        modes_sent.append(mode)
      modes_sents.append(modes_sent)
    return modes_sents

  def get_mode_feature(self, e, mode): 
    if mode == MODE_S:
      return e['head_lemma']
    if mode == MODE_D:
      return e['head_lemma'] 

  def get_vocab(self, entity_feats):
    vocab = {mode: {NULL: 0} for mode in self.modes}
    for e_feat in entity_feats:
      for e_idx, e in enumerate(e_feat):
        for mode in self.modes:
          token = self.get_mode_feature(e, mode)
          if not token in vocab[mode]:
            vocab[mode][token] = 1 
          else:
            vocab[mode][token] += 1        
    return vocab

  def initialize(self):
    translate_prob_file = os.path.join(self.config['model_path'], 'translation_probs.json')
    position_prob_file = os.path.join(self.config['model_path'], 'position_probs.json') 
    if os.path.exists(translate_prob_file):
      self.P_ee = json.load(open(translate_prob_file))
      print(f'Loaded pretrained translation probability file: {translate_prob_file}')
    else:
      for mode in self.modes:
        for v in self.vocab[mode]:
          if not v in self.P_ee[mode]:
            self.P_ee[mode][v] = dict()

          for v2 in self.vocab[mode]:
            if v2 != NULL:
              self.P_ee[mode][v][v2] = 1. / (len(self.vocab[mode]) - 1)          
    
    for mode in self.modes:
      self.P_ij[mode] = dict()
      for e_feat in self.e_feats_train:
        for i in range(len(e_feat)):
          self.P_ij[mode][i + 1] = {j: 1. / (i + 1) for j in range(i + 1)}
      if mode == MODE_D and os.path.exists(position_prob_file):
        print(f'Loaded pretrained position probability file: {position_prob_file}')
        self.P_ij[mode] = json.load(open(position_prob_file)) 

  def is_match(self, e1, e2, mode):
    if mode == MODE_S:
      if e1['head_lemma'] == NULL:
        return True
      v1 = e1['entity_embedding']
      v2 = e2['entity_embedding']
      if cosine_similarity(v1, v2) <= 0.5 or (e1['entity_type'] != e2['entity_type']):
        return False
      if (e1['number'] == SINGULAR or e2['number'] == SINGULAR) and (e1['number'] != e2['number']):
        return False
      if (e1['number'] != PLURAL and e2['number'] != PLURAL) and (e1['number'] != e2['number']):
        return False

    if mode == MODE_D:
      if e1['head_lemma'] == NULL:
        if e2['pos_tag'] in ['PRP', 'PRP$']:
          return False
        else:
          return True
      if e1['entity_type'] != e2['entity_type']:
        return False
      if (e1['number'] == SINGULAR or e2['number'] == SINGULAR) and (e1['number'] != e2['number']):
        return False
      if (e1['number'] != PLURAL and e2['number'] != PLURAL) and (e1['number'] != e2['number']):
        return False
      if e1['pos_tag'] == 'PRP$' and (e2['idx']-e1['idx'] == 1):
        return False
    return True
   
  def compute_alignment_counts(self):
    align_counts = []
    for e_feat, modes_sent in zip(self.e_feats_train, self.modes_sents):
      align_count = {mode: dict() for mode in self.modes}
      for e_idx, (e, mode) in enumerate(zip(e_feat, modes_sent)):
        align_count[mode][e_idx+1] = dict()
        token = self.get_mode_feature(e, mode)

        align_count[mode][e_idx+1][0] = self.P_ij[mode][e_idx+1][0] * self.P_ee[mode][NULL][token]
        if not self.is_match({'head_lemma': NULL}, e, mode):
          align_count[mode][e_idx+1][0] = 0
          
        for a_idx, antecedent in enumerate(e_feat[:e_idx]):
          if self.is_match(e, antecedent, mode):
            a_token = self.get_mode_feature(antecedent, mode)
            align_count[mode][e_idx+1][a_idx+1] = self.P_ij[mode][e_idx+1][a_idx+1] * self.P_ee[mode][a_token][token]
        
        norm_factor = sum(align_count[mode][e_idx+1].values())
        for a_idx in align_count[mode][e_idx+1]: 
          align_count[mode][e_idx+1][a_idx] /= max(norm_factor, EPS)
      align_counts.append(align_count)
    return align_counts

  def update_translation_probs(self):
    P_ee = {mode: dict() for mode in self.modes}
    for e_feat, count, modes_sent in zip(self.e_feats_train, self.ee_counts, self.modes_sents):
      for mode in count:
        for e_idx in count[mode]:
          token = self.get_mode_feature(e_feat[e_idx-1], mode)
          for a_idx in count[mode][e_idx]:
            if a_idx == 0:
              if not NULL in P_ee[mode]:
                P_ee[mode][NULL] = dict()
              
              if not token in P_ee[mode][NULL]:
                P_ee[mode][NULL][token] = count[mode][e_idx][a_idx]  
              else:
                P_ee[mode][NULL][token] += count[mode][e_idx][a_idx]
            else:
              a_token = self.get_mode_feature(e_feat[a_idx-1], mode)
              if not a_token in P_ee[mode]:
                P_ee[mode][a_token] = dict()
              
              if not token in P_ee[mode][a_token]:
                P_ee[mode][a_token][token] = count[mode][e_idx][a_idx]
              else:
                P_ee[mode][a_token][token] += count[mode][e_idx][a_idx] 
          
    # Normalize
    for mode in self.modes:
      for a in P_ee[mode]:
        norm_factor = sum(P_ee[mode][a].values())
        for e in P_ee[mode][a]:
          P_ee[mode][a][e] /= max(norm_factor, EPS) 

    return P_ee

  def update_position_probs(self):
    P_ij = dict()
    for e_feat, count, modes_sent in zip(self.e_feats_train, self.ee_counts, self.modes_sents):
      for mode in count:
        if mode == MODE_D:
          for e_idx in count[mode]:
            if not e_idx in P_ij:
              P_ij[e_idx] = dict()

            for a_idx in count[mode][e_idx]:
              if not a_idx in P_ij[e_idx]:
                P_ij[e_idx][a_idx] = count[mode][e_idx][a_idx]
              else: 
                P_ij[e_idx][a_idx] += count[mode][e_idx][a_idx]
      
    for e_idx in P_ij:
      norm_factor = sum(P_ij[e_idx].values())
      for a_idx in P_ij[e_idx]:
        P_ij[e_idx][a_idx] /= max(norm_factor, EPS) 
    
    return P_ij

  def log_likelihood(self):
    ll = 0.
    for ex, (e_feat, modes_sent) in enumerate(zip(self.e_feats_train, self.modes_sents)):
      for e_idx, (e, mode) in enumerate(zip(e_feat, modes_sent)):
        token = self.get_mode_feature(e, mode)
        probs = [self.P_ij[mode][e_idx+1][0] * self.P_ee[mode][NULL][token]]
        for a_idx, a in enumerate(e_feat[:e_idx]):
          if self.is_match(e, a, mode):
            a_token = self.get_mode_feature(a, mode)
            probs.append(self.P_ij[mode][e_idx+1][a_idx+1] * self.P_ee[mode][a_token][token])
        ll += np.log(np.maximum(np.mean(probs), EPS))
    return ll

  def train(self, n_epochs=10):
    for epoch in range(n_epochs):
      self.ee_counts = self.compute_alignment_counts()
      self.P_ee = self.update_translation_probs()
      self.P_ij[MODE_D] = self.update_position_probs()
      json.dump(self.P_ij, open(os.path.join(self.config['model_path'], 'position_probs.json'), 'w'), indent=2)
      json.dump(self.P_ee, open(os.path.join(self.config['model_path'], 'translation_probs.json'), 'w'), indent=2)
      logging.info('Epoch {}, log likelihood = {:.3f}'.format(epoch, self.log_likelihood()))           
      print('Epoch {}, log likelihood = {:.3f}'.format(epoch, self.log_likelihood()))
 
  def predict_antecedents(self, entity_features):
    antecedents = []
    cluster_ids = []
    n_cluster = 0
    modes_sents = self.get_modes(entity_features)
    for e_feat, modes_sent in zip(entity_features, modes_sents):
      antecedent = [-1]*len(e_feat)
      cluster_id = [0]*len(e_feat)
      for e_idx, (e, mode) in enumerate(zip(e_feat, modes_sent)):
        token = self.get_mode_feature(e, mode)
        scores = [self.P_ij[mode][e_idx+1][0] * self.P_ee[mode][NULL][token]]
        for a_idx, a in enumerate(e_feat[:e_idx]):
          if self.is_match(e, a, mode):
            a_token = self.get_mode_feature(a, mode)
            scores.append(self.P_ij[mode][e_idx+1][a_idx+1] * self.P_ee[mode][a_token][token])
          else:
            scores.append(0)
        scores = np.asarray(scores)
        antecedent[e_idx] = np.argmax(scores) - 1
        if antecedent[e_idx] == -1:
          n_cluster += 1
          cluster_id[e_idx] = n_cluster
        else:
          cluster_id[e_idx] = cluster_id[antecedent[e_idx]]
      antecedents.append(antecedent)
      cluster_ids.append(cluster_id)
    return antecedents, cluster_ids

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
 
def load_text_features(config, split):
  lemmatizer = WordNetLemmatizer()
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
      
      entities.append(entity)  
    cluster_ids = [label_dict[span]['cluster_id'] for span in spans]
    
    entity_feats.append(entities)
    doc_ids.append(doc_id)
    spans_all.append(spans)
    cluster_ids_all.append(np.asarray(cluster_ids))
    tokens_all.append([t[2] for t in doc[doc_id]])
  return entity_feats,\
         doc_ids,\
         spans_all,\
         cluster_ids_all,\
         tokens_all

def load_data(config):
  lemmatizer = WordNetLemmatizer() 
  entity_mentions_train = []
  doc_train = dict()
  for split in config['splits']['train']:
    entity_mentions_train.extend(json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_entities.json'), 'r', 'utf-8')))
    doc_train.update(json.load(codecs.open(os.path.join(config['data_folder'], f'{split}.json'))))

  entity_mentions_test = []
  doc_test = dict()
  for split in config['splits']['test']:
    entity_mentions_test.extend(json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_entities.json'), 'r', 'utf-8')))
    doc_test.update(json.load(codecs.open(os.path.join(config['data_folder'], f'{split}.json'))))

  entity_feats_train = []
  doc_ids_train = []
  spans_train = []
  cluster_ids_train = []
  tokens_train = []
  for split in config['splits']['train']:
    cur_entity_feats_train,\
    cur_doc_ids_train,\
    cur_spans_train,\
    cur_cluster_ids_train,\
    cur_tokens_train = load_text_features(config, split=split)
    
    entity_feats_train.extend(cur_entity_feats_train)
    doc_ids_train.extend(cur_doc_ids_train)
    spans_train.extend(cur_spans_train)
    cluster_ids_train.extend(cur_cluster_ids_train)
    tokens_train.extend(cur_tokens_train)
  print(f'Number of training examples: {len(entity_feats_train)}')
  
  entity_feats_test = []
  doc_ids_test = []
  spans_test = []
  cluster_ids_test = []
  tokens_test = []
  for split in config['splits']['test']:
    cur_entity_feats_test,\
    cur_doc_ids_test,\
    cur_spans_test,\
    cur_cluster_ids_test,\
    cur_tokens_test = load_text_features(config, split='test')
    
    entity_feats_test.extend(cur_entity_feats_test)
    doc_ids_test.extend(cur_doc_ids_test)
    spans_test.extend(cur_spans_test)
    cluster_ids_test.extend(cur_cluster_ids_test)
    tokens_test.extend(cur_tokens_test)
  print(f'Number of test examples: {len(entity_feats_test)}')

  return entity_feats_train,\
         doc_ids_train,\
         spans_train,\
         cluster_ids_train,\
         tokens_train,\
         entity_feats_test,\
         doc_ids_test,\
         spans_test,\
         cluster_ids_test,\
         tokens_test,\

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', '-c', default='../configs/config_text_smt_video_m2e2.json')
  args = parser.parse_args()

  config_file = args.config
  config = pyhocon.ConfigFactory.parse_file(config_file)
  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  logging.basicConfig(filename=os.path.join(config['model_path'], 'train.log'), level=logging.DEBUG)

  entity_feats_train,\
  doc_ids_train,\
  spans_train,\
  cluster_ids_train,\
  tokens_train,\
  entity_feats_test,\
  doc_ids_test,\
  spans_test,\
  cluster_ids_test,\
  tokens_test = load_data(config)

  ## Model training
  aligner = SMTEntityCoreferencer(entity_feats_train+entity_feats_test, config)
  aligner.train(10)
  _, cluster_ids_all = aligner.predict_antecedents(entity_feats_test)

  ## Test and evaluation
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
