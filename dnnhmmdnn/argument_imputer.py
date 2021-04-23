import numpy as np
import json
import os
from copy import deepcopy
import logging
import pyhocon
import codecs
import argparse 
NULL = '###NULL###'
EPS = 1e-100
class MixtureArgumentImputer:
  def __init__(self, event_features,
               argument_types):
    """
    Attributes:
        event_features: a list of list of dict from feature type to a
          str token, where the token is NULL if its value is missing  
        argument_feature_types: a list of feature types for arguments
    """
    self.event_features = event_features
    self.argument_types = argument_types
    self.ea_counts = [] # list[dict[tuple[int, str], dict[tuple[int, str], float]]]]
    self.P_ea = dict() # dict[str, dict[str, dict[str, float]]]

    self.vocab = self.get_vocab(event_features) # dict[str, dict[str, int]]
    for feat_type in self.vocab:
      print(f'{feat_type} vocab size: {len(self.vocab[feat_type])}')
    self.initialize()

    json.dump(self.vocab, open('argument_imputer_vocabs.json', 'w'))

  def initialize(self):
    for a_type in self.argument_types:
      if not a_type in self.P_ea:
        self.P_ea[a_type] = dict()

      for trigger in self.vocab['trigger']:
        if not trigger in self.P_ea:
          self.P_ea[a_type][trigger] = dict()
        
        for a in self.vocab['argument']:
          self.P_ea[a_type][trigger][a] = 1 ./ len(self.vocab['argument']) 

  def get_vocab(self, event_feats):
    vocab = {'trigger': dict(), 'argument': dict()}
    for e_feat in event_feats:
      for e_idx, e in enumerate(e_feat):
        if not e['head_lemma'] in vocab['trigger']:
          vocab['trigger'][e['head_lemma']] = 1
        else:
          vocab['trigger'][e['head_lemma']] += 1

        a_types = self.get_arg_types(e)

        for a_type in a_types:
          if not e[a_type] in vocab['argument']:
            vocab['argument'][e[a_type]] = 1
          else:
            vocab['argument'][e[a_type]] += 1

          if not a_type in vocab:
            vocab[a_type] = dict()
            vocab[a_type][e[a_type]] = 1
          elif not e[a_type] in vocab[a_type]:
            vocab[a_type][e[a_type]] = 1
          else:
            vocab[a_type][e[a_type]] += 1
    total_args = sum(vocab['argument'].values())
    num_miss_args = vocab['argument'].get(NULL, 0)
    print(f'{num_miss_args} of {total_args} arguments are missing values')
    return vocab

  def get_arg_types(self, e):
    a_types = []
    for feat_type in e:
      if feat_type in self.argument_types:
        a_types.append(feat_type)
    return a_types

  def compute_alignment_counts(self):
    ea_counts = []
    for e_feat in self.event_features:
      ea_count = dict()
      for e_idx, e in enumerate(e_feat):
        trigger = e['head_lemma']
        a_types = self.get_a_types(e)
        for a_type in a_types:
          ea_count[(e_idx, a_type)] = dict()

          a = e[feat_type]
          if a == NULL: # Impute the data
            for e_idx2, e2 in enumerate(e_feat):
              if e_idx == e_idx2:
                continue
              a_types2 = self.get_a_types(e2)
              for a_type2 in a_types2:
                a2 = e2[a_type2]
                ea_count[(e_idx, a_type)][(e_idx2, a_type2)] = self.P_ea[a_type][trigger][a2]
            
            if len(ea_count[(e_idx, a_type)]) == 0: # If there is only one event, keep the empty entry
              ea_count[(e_idx, a_type)][(e_idx, a_type)] = 1.
                
            # Normalize
            for k in ea_count[(e_idx, a_type)]:
              ea_count[(e_idx, a_type)][k] /= sum(ea_count[(e_idx, a_type)].values()) 
          else:
            ea_count[(e_idx, a_type)][(e_idx, a_type)] = 1.
      ea_counts.append(ea_count)
    return ea_counts

  def update_translation_probs(self):
    P_ea = dict()
    for e_feat, ea_count in zip(self.event_features, self.ea_counts):
      for e_idx, a_type in ea_count:
        for (e_idx2, a_type2), count in ea_count[(e_idx, a_type)].items():
          trigger = e_feat[e_idx]['head_lemma']
          arg = e_feat[e_idx2][a_type2]
          if not a_type in P_ea:
            P_ea[a_type] = dict()

          if not trigger in P_ea[a_type]:
            P_ea[a_type][trigger] = dict()
            
          if not arg in P_ea[a_type][trigger]: 
            P_ea[a_type][trigger][arg] = count
          else:
            P_ea[a_type][trigger][arg] += count 

    # Normalize
    for a_type in P_ea:
      for trigger in P_ea[a_type]:
        for arg in P_ea[a_type][trigger]:
          P_ea[a_type][trigger][arg] /= sum(P_ea[a_type][trigger].values())
    return P_ea

  def log_likelihood(self):
    ll = 0.
    for ex, e_feat in enumerate(self.event_features):
      for e_idx in enumerate(e_feat):
        trigger = e['head_lemma']
        a_types = self.get_arg_types(e)
        for a_type in a_types:
          a = e[a_type]
          if a == 'NULL':
            probs = []
            for e_idx2, e2 in enumerate(e_feat):
              if e_idx2 == e_idx:
                continue
              a_types2 = self.get_arg_types(e2)
              for a_type2 in a_types2:
                a2 = e2[a_type2]
                probs.append(self.P_ea[a_type][trigger][a2])
            ll += np.log(np.maximum(np.mean(probs), EPS))
          else:
            ll += np.log(np.maximum(self.P_ea[a_type][trigger][a], EPS))
    return ll

  def trainEM(self, n_iter=10):
    for epoch in range(n_iter):
      self.ea_counts = self.compute_alignment_counts()
      self.P_ea = self.update_translation_probs()
      print(f'Iteration {epoch}, log likelihood: {self.log_likelihood()}')

  def impute(self, event_features):
    for event_feat in event_features:
      for e_idx, e in enumerate(event_feat):
        trigger = e['head_lemma']
        a_types = self.get_arg_types(e)
        for a_type in a_types:
          a = e[a_type]
          if a == 'NULL':
            probs = [] 
            candidates = []
            for e_idx2, e2 in enumerate(e_feat):
              if e_idx == e_idx2:
                continue
              a_types2 = self.get_arg_types(e2)
              for a_type2 in a_types2:
                a2 = e2[a_type2]
                probs.append(self.P_ea[a_type][trigger][a2])
                candidates.append(a2)

            if len(probs) > 0:
              best_idx = np.argmax(probs)
              e[a_type] = candidates[best_idx]
    return event_features

def load_data(config):
  event_mentions_train = json.load(codecs.open(os.path.join(config['data_folder'], 'train_events_with_linguistic_features.json'), 'r', 'utf-8'))
  event_mentions_test = json.load(codecs.open(os.path.join(config['data_folder'], 'test_events_with_linguistic_features.json'), 'r', 'utf-8'))
  
  ontology = dict()
  for m in event_mentions_train+event_mentions_test:
    event_type = m['event_type']
    if not event_type in ontology:
      ontology[event_type] = []
      for a in m['arguments']:
        if not a['role'] in ontology[event_type]:
          ontology[event_type].append(a['role'])
  arg_types = [a for e in ontology for a in ontology[e]]
  json.dump(ontology, open(os.path.join(config['data_folder'], 'text_ontology.json'), 'w'), indent=2)

  doc_ids_train = []
  for m in event_mentions_train:
    if not m['doc_id'] in label_dicts:
      label_dicts[m['doc_id']] = {}
      doc_ids_train.append(m['doc_id'])
    span = (min(m['tokens_ids']), max(m['tokens_ids']))
    new_m = dict()
    for k in m:
      trigger = m['head_lemma']
      event_type = m['event_type']
      if k == 'arguments':
        for a_type in ontology[event_type]:
          new_m[a_type] = NULL 
        
        for a in m['arguments']:
          new_m[a['role']] = a['head_lemma']
      else:
        new_m[k] = deepcopy(m[k])
    label_dicts[m['doc_id']][span] = deepcopy(new_m)

  doc_ids_test = []
  for m in event_mentions_test:
    if not m['doc_id'] in label_dicts:
      label_dicts[m['doc_id']] = {}
      doc_ids_test.append(m['doc_id'])
    span = (min(m['tokens_ids']), max(m['tokens_ids']))
    new_m = dict()
    for k in m:
      trigger = m['head_lemma']
      event_type = m['event_type']
      if k == 'arguments':
        for a_type in ontology[event_type]:
          new_m[a_type] = NULL 
        
        for a in m['arguments']:
          new_m[a['role']] = a['head_lemma']
      else:
        new_m[k] = deepcopy(m[k])
    label_dicts[m['doc_id']][span] = deepcopy(new_m)

  event_features_train = []
  for doc_id in doc_ids_train[:20]: # XXX
    e_feat = [label_dicts[doc_id][span] for span in sorted(label_dicts[doc_id])]
    event_features_train.append(e_feat)

  event_features_test = []
  for doc_id in doc_ids_test[:20]: # XXX
    e_feat = [label_dicts[doc_id][span] for span in sorted(label_dicts[doc_id])]
    event_features_test.append(e_feat)

  return event_features_train, event_features_test, arg_types

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', 'c')
  args = parser.parse_args()

  config_file = args.config
  config = pyhocon.ConfigFactory.parse_file(config_file)
  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  logging.basicConfig(filename=os.path.join(config['model_path'], 'train.log'), level=logging.DEBUG)

  event_feats_train, event_feats_test, argument_types = load_data(config)
  imputer = MixtureArgumentImputer(event_feats_train+event_feats_test, argument_types)
  new_event_feats_train = imputer.impute(event_feats_train)
  new_event_feats_train = [e for e_feat in new_event_feats_train for e in e_feat]
  new_event_feats_test = imputer.impute(event_feats_test)
  new_event_feats_test = [e for e_feat in new_event_feats_test for e in e_feat]

  json.dump(new_event_feats_train, open(os.path.join(config['model_path'], 'train_events_imputed.json'), 'w'), indent=2)
  json.dump(new_event_feats_test, open(os.path.join(config['model_path'], 'test_events_imputed.json'), 'w'), indent=2)
