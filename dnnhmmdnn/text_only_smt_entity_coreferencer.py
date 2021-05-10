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
from evaluator import Evaluation, CoNLLEvaluation
np.random.seed(2)
random.seed(2)

NULL = '###NEW###'
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
    self.P_ee = dict()
    self.vocab = self.get_vocab(entity_features)
    self.initialize()
    logging.info(f'Number of documents = {len(self.e_feats_train)}, vocab size = {len(self.vocab)}')

  def get_vocab(self, entity_feats):
    vocab = {NULL:0}
    for e_feat in entity_feats:
      for e_idx, e in enumerate(e_feat):
        if not e['head_lemma'] in vocab:
          vocab[e['head_lemma']] = 1 
        else:
          vocab[e['head_lemma']] += 1 
    return vocab

  def initialize(self):
    for v in self.vocab:
      if not v in self.P_ee:
        self.P_ee[v] = dict()

      for v2 in self.vocab:
        if v2 != NULL:
          self.P_ee[v][v2] = 1. / (len(self.vocab) - 1)   

  def is_match(self, e1, e2):
    v1 = e1['entity_embedding']
    v2 = e2['entity_embedding']
    if cosine_similarity(v1, v2) <= 0.5:
      return False

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
   
  def compute_alignment_counts(self):
    align_counts = []
    for e_feat in self.e_feats_train:
      align_count = {0: dict()}
      for e_idx, e in enumerate(e_feat):
        align_count[e_idx+1] = dict()
        token = e['head_lemma']

        align_count[e_idx+1][0] = self.P_ee[NULL][token]
        for a_idx, antecedent in enumerate(e_feat[:e_idx]):
          if self.is_match(e, antecedent):
            a_token = antecedent['head_lemma']
            align_count[e_idx+1][a_idx+1] = self.P_ee[a_token][token]
        
        norm_factor = sum(align_count[e_idx+1].values())
        for a_idx in align_count[e_idx+1]: 
          align_count[e_idx+1][a_idx] /= norm_factor
      align_counts.append(align_count)
    return align_counts

  def update_translation_probs(self):
    P_ee = dict()
    for e_feat, count in zip(self.e_feats_train, self.ee_counts):
      for e_idx in count:
        token = e_feat[e_idx-1]['head_lemma']
        for a_idx in count[e_idx]:
          if a_idx == 0:
            if not NULL in P_ee:
              P_ee[NULL] = dict()
            
            if not token in P_ee[NULL]:
              P_ee[NULL][token] = count[e_idx][a_idx]  
            else:
              P_ee[NULL][token] += count[e_idx][a_idx]  
          else:
            a_token = e_feat[a_idx-1]['head_lemma']
            if not a_token in P_ee:
              P_ee[a_token] = dict()
            
            if not token in P_ee[a_token]:
              P_ee[a_token][token] = count[e_idx][a_idx]
            else:
              P_ee[a_token][token] += count[e_idx][a_idx] 
          
    # Normalize
    for a in P_ee:
      norm_factor = sum(P_ee[a].values())
      for e in P_ee[a]:
        P_ee[a][e] /= norm_factor   

    return P_ee

  def log_likelihood(self):
    ll = 0.
    for ex, e_feat in enumerate(self.e_feats_train):
      for e_idx, e in enumerate(e_feat):
        probs = [self.P_ee[NULL][e['head_lemma']]]
        for a_idx, a in enumerate(e_feat[:e_idx]):
          if self.is_match(e, a):
            probs.append(self.P_ee[a['head_lemma']][e['head_lemma']])
        ll += np.log(np.maximum(np.mean(probs), EPS))
    return ll

  def train(self, n_epochs=10):
    for epoch in range(n_epochs):
      self.ee_counts = self.compute_alignment_counts()
      self.P_ee = self.update_translation_probs()
      json.dump(self.P_ee, open(os.path.join(self.config['data_folder'], 'translation_probs.json'), 'w'), indent=2)
      logging.info('Epoch {}, log likelihood = {:.3f}'.format(epoch, self.log_likelihood()))           
      print('Epoch {}, log likelihood = {:.3f}'.format(epoch, self.log_likelihood()))
 
  def predict_antecedents(self, entity_features):
    antecedents = []
    cluster_ids = []
    n_cluster = 0
    for e_feat in entity_features:
      antecedent = [-1]*len(e_feat)
      cluster_id = [0]*len(e_feat)
      for e_idx, e in enumerate(e_feat):
        scores = [self.P_ee[NULL][e['head_lemma']]]
        for a_idx, a in enumerate(e_feat[:e_idx]):
          if self.is_match(e, a):
            scores.append(self.P_ee[a['head_lemma']][e['head_lemma']])
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
 
def load_text_features(config, vocab_feat, split):
  lemmatizer = WordNetLemmatizer()
  feature_types = config['feature_types']
  entity_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_entities.json'), 'r', 'utf-8'))
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}.json')))
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
        label_dicts[m['doc_id']] = {}
      token = lemmatizer.lemmatize(m['tokens'].lower(), pos='v')
      span = (min(m['tokens_ids']), max(m['tokens_ids']))
      label_dicts[m['doc_id']][span] = {'token_id': token,
                                        'cluster_id': m['cluster_id']} # vocab_feat['entity_type'][m['entity_type']]} # XXX  

      for feat_type in feature_types:
        label_dicts[m['doc_id']][span][feat_type] = m[feat_type] 
        
  for feat_idx, doc_id in enumerate(sorted(label_dicts)): # XXX
    doc_embs = docs_embs[doc_to_feat[doc_id]]
    label_dict = label_dicts[doc_id]
    spans = sorted(label_dict)
    entities = []
    for span_idx, span in enumerate(spans):
      entity = {feat_type: label_dict[span][feat_type] for feat_type in feature_types}
      entity['entity_embedding'] = doc_embs[span_idx, :300]
      entities.append(entity)  
    cluster_ids = [label_dict[span]['cluster_id'] for span in spans]
    
    entity_feats.append(entities)
    doc_ids.append(doc_id)
    spans_all.append(spans)
    cluster_ids_all.append(np.asarray(cluster_ids))
    tokens_all.append([t[2] for t in doc_train[doc_id]])
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

  feature_types = config['feature_types']
  vocab_feats = {feat_type:{NULL: 0} for feat_type in feature_types}
  vocab_feats_freq = {feat_type:{NULL: 0} for feat_type in feature_types}
  for m in entity_mentions_train + entity_mentions_test:    
    for feat_type in feature_types: 
      if not m[feat_type] in vocab_feats[feat_type]:
        vocab_feats[feat_type][m[feat_type]] = len(vocab_feats[feat_type])
        vocab_feats_freq[feat_type][m[feat_type]] = 1
      else:
        vocab_feats_freq[feat_type][m[feat_type]] += 1
  json.dump(vocab_feats_freq, open('vocab_feats_freq.json', 'w'), indent=2)

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
    cur_tokens_train = load_text_features(config, vocab_feats_freq, split=split)
    
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
    cur_tokens_test = load_text_features(config, vocab_feats_freq, split='test')
    
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
         vocab_feats

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
  tokens_test,\
  vocab_feats = load_data(config)

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
