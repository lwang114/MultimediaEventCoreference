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
np.random.seed(2)
random.seed(2)

NULL = '###NEW###'
EPS = 1e-100
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
    self.e_feats_train = event_features
    self.v_feats_train = action_features

    self.ev_counts = dict()
    self.ee_counts = dict()
    self.centroids = None
    self.vars = None
    self.P_ve = dict()
    self.P_ee = dict()
    self.Kv = config.get('Kv', 4)
    self.vocab = self.get_vocab(event_features)

    self.initialize()
    logging.info(f'Number of documents = {len(self.e_feats_train)}, vocab.size = {len(self.vocab)}')
    
  def get_vocab(self, event_feats):
    vocab = {NULL:0}
    for e_feat in event_feats:
      for e_idx, e in enumerate(e_feat):
        if not e['head_lemma'] in vocab:
          vocab[e['head_lemma']] = 1
        else:
          vocab[e['head_lemma']] += 1
    return vocab 
  
  def initialize(self):
    # Initialize event-event translation probs
    for v in self.vocab:
      self.P_ee[v] = {v2: 1. / (len(self.vocab) - 1) for v2 in self.vocab if v2 != NULL}

    # Initialize action-event translation probs
    for k in range(self.Kv):
      self.P_ve[k] = {v: 1. / (len(self.vocab) - 1) for v in self.vocab if v != NULL} 

    # Initialize action cluster centroids
    X = np.concatenate(self.v_feats_train)
    kmeans = KMeans(n_clusters=self.Kv).fit(X)

    self.centroids = deepcopy(kmeans.cluster_centers_)
    y = kmeans.predict(X)
    self.vars = EPS * np.ones(self.Kv)
    for k in range(self.Kv):
      self.vars[k] = np.var(X[(y == k).nonzero()[0]]) * X.shape[1] / 3

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
  
  def is_match(self, e1, e2, print_score=False): # XXX
    v1 = e1['trigger_embedding']
    v2 = e2['trigger_embedding']
    if cosine_similarity(v1, v2) <= 0.5:
      return False 
    if not self.has_same_plurality(e1, e2):
      return False 
    return True 
  
  def compute_alignment_counts(self):
    ev_counts = []
    ee_counts = []
    for e_feat, v_feat in zip(self.e_feats_train, self.v_feats_train):
      C_ee = dict()
      C_ev = dict()
      L = v_feat.shape[0]
      v_prob = self.compute_cluster_prob(v_feat)

      for e_idx, e in enumerate(e_feat):
        C_ee[e_idx] = dict()
        C_ev[e_idx] = dict()
        token = e['head_lemma']

        # Compute event-event alignment counts
        C_ee[e_idx][0] = self.P_ee[NULL][token] 
        for a_idx, a in enumerate(e_feat[:e_idx]):
          if self.is_match(e, a):
            a_token = a['head_lemma']
            C_ee[e_idx][a_idx+1] = self.P_ee[a_token][token]
        
        # Compute event-action alignment counts
        if e.get('is_visual', True):
          e_prob = np.asarray([self.P_ve[k][token] for k in range(self.Kv)])
          for v_idx, v in enumerate(v_feat):
            C_ev[e_idx][v_idx] = v_prob[v_idx] * e_prob 

        norm_factor = sum(C_ee[e_idx].values())
        norm_factor += sum(C_ev[e_idx][v_idx].sum() for v_idx in C_ev[e_idx])

        for a_idx in C_ee[e_idx]:
          C_ee[e_idx][a_idx] /= norm_factor
        
        for v_idx in C_ev[e_idx]:
          C_ev[e_idx][v_idx] /= norm_factor
          C_ev[e_idx][v_idx] = C_ev[e_idx][v_idx].tolist()
        
      ee_counts.append(C_ee)
      ev_counts.append(C_ev)

    return ee_counts, ev_counts 

  def update_translation_probs(self):
    P_ee = dict()
    P_ve = {k:dict() for k in range(self.Kv)}
    for e_feat, v_feat, ee_count, ev_count in zip(self.e_feats_train, self.v_feats_train, self.ee_counts, self.ev_counts):
      for e_idx in ee_count:
        token = e_feat[e_idx]['head_lemma']
        for a_idx in ee_count[e_idx]:
          if a_idx == 0:
            if not NULL in P_ee:
              P_ee[NULL] = dict()
            
            if not token in P_ee[NULL]:
              P_ee[NULL][token] = ee_count[e_idx][a_idx]
            else:
              P_ee[NULL][token] += ee_count[e_idx][a_idx]
          else:
            a_token = e_feat[a_idx-1]['head_lemma']
            if not a_token in P_ee:
              P_ee[a_token] = dict()

            if not token in P_ee[a_token]:
              P_ee[a_token][token] = ee_count[e_idx][a_idx]
            else:
              P_ee[a_token][token] += ee_count[e_idx][a_idx]
        
        for v_idx in ev_count[e_idx]:
          for k in range(self.Kv):
            if not token in P_ve[k]:
              P_ve[k][token] = ev_count[e_idx][v_idx][k]
            else:
              P_ve[k][token] += ev_count[e_idx][v_idx][k]

    # Normalize
    for a in P_ee:
      norm_factor = sum(P_ee[a].values())
      for e in P_ee[a]:
        P_ee[a][e] /= norm_factor
    
    for v in P_ve:
      norm_factor = sum(P_ve[v].values())
      for e in P_ve[v]:
        P_ve[v][e] /= norm_factor
    return P_ee, P_ve

  def log_likelihood(self):
    ll = 0.
    for e_feat, v_feat in zip(self.e_feats_train, self.v_feats_train):
      v_prob = self.compute_cluster_prob(v_feat)
      for e_idx, e in enumerate(e_feat):
        probs = [self.P_ee[NULL][e['head_lemma']]]
        for a_idx, a in enumerate(e_feat[:e_idx]):
          if self.is_match(e, a):
            probs.append(self.P_ee[a['head_lemma']][e['head_lemma']])  
        
        if e.get('is_visual', True):
          e_prob = np.asarray([self.P_ve[k][e['head_lemma']] for k in range(self.Kv)])
          probs.extend((v_prob @ e_prob).tolist())
        
        ll += np.log(np.maximum(np.mean(probs), EPS))
    return ll
           
  def train(self, n_epochs=10):
    for epoch in range(n_epochs):
      self.ee_counts, self.ev_counts = self.compute_alignment_counts()
      self.P_ee, self.P_ve = self.update_translation_probs()
                 
      json.dump(self.P_ee, open(os.path.join(self.config['model_path'], 'ee_translation_probs.json'), 'w'), indent=2)
      json.dump(self.P_ve, open(os.path.join(self.config['model_path'], 've_translation_probs.json'), 'w'), indent=2)

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

  def predict_antecedents(self, event_features, action_features):
    antecedents = []
    cluster_ids = []
    scores_all = []

    n_cluster = 0
    for e_feat, v_feat in zip(event_features, action_features):
      v_prob = self.compute_cluster_prob(v_feat)
      antecedent = [-1]*len(e_feat)
      cluster_id = [0]*len(e_feat)
      scores_all.append([])
      L = v_feat.shape[0]
      for e_idx, e in enumerate(e_feat):
        e_token = e['head_lemma']
        e_prob = self.action_event_prob(v_feat, e) # TODO
        scores.append(self.P_ee[NULL][e_token])
        for a_idx, a in enumerate(e_feat[:e_idx]):
          a_token = a['head_lemma']
          if self.is_match(e, a, mode='text'): # TODO
            scores.append(self.P_ee[a_token][e_token])
          elif self.is_match(e, a, mode='visual'):
            a_prob = self.action_event_prob(v_feat, a)
            scores.append(v_prob @ (e_prob * a_prob).mean(0))
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

 
def load_text_features(config, vocab_feat, action_labels, split):
  lemmatizer = WordNetLemmatizer()
  feature_types = config['feature_types']
  event_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_events.json'), 'r', 'utf-8'))
  entity_mentions = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}_entities.json'), 'r', 'utf-8'))
  ontology_map = json.load(open(os.path.join(config['data_folder'], '../ontology_mapping.json')))
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], f'{split}.json')))
  docs_embs = np.load(os.path.join(config['data_folder'], f'{split}_events_with_arguments_glove_embeddings.npz')) 
  doc_to_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in docs_embs}

  label_dicts = dict()
  event_feats = []
  doc_ids = []
  spans_all = []
  cluster_ids_all = []
  tokens_all = [] 
  entity_label_dicts = dict()
  for m in entity_mentions:
    if not m['doc_id'] in entity_label_dicts:
      entity_label_dicts[m['doc_id']] = dict()
    span = (min(m['tokens_ids']), max(m['tokens_ids']))
    entity_label_dicts[m['doc_id']][span] = m['cluster_id']

  for m in event_mentions:
      if not m['doc_id'] in label_dicts:
        label_dicts[m['doc_id']] = dict()
      token = lemmatizer.lemmatize(m['tokens'].lower(), pos='v')
      span = (min(m['tokens_ids']), max(m['tokens_ids']))
      label_dicts[m['doc_id']][span] = {'token_id': token,
                                        'cluster_id': m['cluster_id']}

      for feat_type in feature_types:
        label_dicts[m['doc_id']][span][feat_type] = m[feat_type]
        if feat_type == 'arguments':
          for a_idx, a in enumerate(label_dicts[m['doc_id']][span]['arguments']):
            a_span = (a['start'], a['end'])
            a_cluster_id = entity_label_dicts[m['doc_id']][a_span]
            label_dicts[m['doc_id']][span]['arguments'][a_idx]['cluster_id'] = a_cluster_id  
        
  for feat_idx, doc_id in enumerate(sorted(label_dicts)): # XXX
    action_label = action_labels[feat_idx]
    doc_embs = docs_embs[doc_to_feat[doc_id]]
    label_dict = label_dicts[doc_id]
    spans = sorted(label_dict)
    events = []
    for span_idx, span in enumerate(spans):
      event = {feat_type: label_dict[span][feat_type] for feat_type in feature_types}
      event['trigger_embedding'] = doc_embs[span_idx, :300]
      event['argument_embedding'] = doc_embs[span_idx, 300:].reshape(-1, 300)
      if 'event_type' in event:
        match_action_types = ontology_map[event['event_type']]
        if len(set(match_action_types).intersection(set(action_label))) > 0:
          event['is_visual'] = True
        else:
          event['is_visual'] = False
      events.append(event)  
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
  lemmatizer = WordNetLemmatizer()
  event_mentions_train = json.load(codecs.open(os.path.join(config['data_folder'], 'train_events.json'), 'r', 'utf-8'))
  doc_train = json.load(codecs.open(os.path.join(config['data_folder'], 'train.json')))
  event_mentions_test = json.load(codecs.open(os.path.join(config['data_folder'], 'test_events.json'), 'r', 'utf-8'))
  doc_test = json.load(codecs.open(os.path.join(config['data_folder'], 'test.json')))
  feature_types = config['feature_types']

  vocab_feats = {feat_type: {NULL: 0} for feat_type in feature_types}
  vocab_feats_freq = {feat_type: {NULL: 0} for feat_type in feature_types}
  for m in event_mentions_train + event_mentions_test:    
    for feat_type in feature_types: 
      if feat_type == 'arguments':
        for a in m['arguments']:
          if not a['head_lemma'] in vocab_feats[feat_type]:
            vocab_feats['arguments'][a['head_lemma']] = 1
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

  action_feats_train, action_labels_train = load_visual_features(config, split='train')
  event_feats_train,\
  doc_ids_train,\
  spans_train,\
  cluster_ids_train,\
  tokens_train = load_text_features(config, vocab_feats_freq, action_labels_train, split='train')
  print(f'Number of training examples: {len(event_feats_train)}')
  
  action_feats_test, action_labels_test = load_visual_features(config, split='test')
  event_feats_test,\
  doc_ids_test,\
  spans_test,\
  cluster_ids_test,\
  tokens_test = load_text_features(config, vocab_feats_freq, action_labels_test, split='test')

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
         tokens_test,\
         vocab_feats

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
  tokens_test,\
  vocab_feats = load_data(config)

  ## Model training
  aligner = AmodalSMTCoreferencer(event_feats_train+event_feats_test, action_feats_train+action_feats_test, config)
  aligner.train(10)
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
