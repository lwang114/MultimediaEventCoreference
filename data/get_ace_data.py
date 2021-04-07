# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import os
import collections
import argparse
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.structured_prediction
import nltk

# dep_parser = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz')

def get_mention_doc(data_json, out_prefix):
  '''
  :param data_json: str of filename of the meta info file storing a json list of dicts with keys:
      doc_id: str,
      sent_id: str,
      tokens: list of str,
      entity_mentions: list of dicts of
        id: str,
        text: str,
        start: int, start token idx 
        end: int, end token idx
      event_mentions: list of dicts of
        id: str,
        text: str,
        start: int, start token idx
        end: int, end token idx

  :param out_prefix: str of the prefix of three files:
      1. [prefix].json: dict of 
        [doc_id]: list of [sent id, token id, token, is entity/event]
      2,3. [prefix]_[entities/events].json: store list of dicts of:
      {'doc_id': str, file prefix of the image for the caption,
       'subtopic': '0',
       'm_id': '0',
       'sentence_id': str, order of the sentence,
       'tokens_ids': list of ints, one-indexed position of the tokens of the current mention in the sentence,
       'tokens': str, tokens concatenated with space,
       'cluster_id': int, 
       'tags': '',
       'lemmas': '',
       'singleton': boolean, whether the mention is a singleton} 
  '''
  documents = {}
  entities = []
  events = []
  sen_start = 0
  
  cluster_ids = {'###SINGLETON###': 0}
  cluster_count = {'###SINGLETON###': 0}
  with open(data_json, 'r') as f:
    # Extract cluster ids
    for line in f:
      inst = json.loads(line)
      entity_mentions = inst['entity_mentions']
      event_mentions = inst['event_mentions']

      for entity_mention in entity_mentions:
        mention_id = entity_mention['id']
        entity_id = '-'.join(mention_id.split('-')[:-1])
        if not entity_id in cluster_count:
          cluster_count[entity_id] = 1
          cluster_ids[entity_id] = 0
        else:
          cluster_count[entity_id] += 1
          cluster_ids[entity_id] = len(cluster_ids)
      
      for event_mention in event_mentions:
        mention_id = event_mention['id']
        event_id = '-'.join(mention_id.split('-')[:-1])
        if not event_id in cluster_count:
          cluster_count[event_id] = 1
          cluster_ids[event_id] = 0
        else:
          cluster_count[event_id] += 1
          cluster_ids[event_id] = len(cluster_ids)
  
  with open(data_json, 'r') as f:
    i = 0
    for line in f:
      # if i > 20:
      #   break
      i += 1
      inst = json.loads(line)
      doc_id = inst['doc_id']
      sent_id = inst['sent_id']
      tokens = inst['tokens']
      pos_tags = [tp[1] for tp in nltk.pos_tag(tokens, tagset='universal')]
      # instance = dep_parser._dataset_reader.text_to_instance(tokens, pos_tags)
      # parsed_sent = dep_parser.predict_instance(instance)
      # dep_role = parsed_sent['predicted_dependencies']
      # dep_head = [h_idx-1 for h_idx in parsed_sent['predicted_heads']]

      entity_mentions = inst['entity_mentions']
      event_mentions = inst['event_mentions']

      if not doc_id in documents:
        documents[doc_id] = []
        sen_start = 0

      entity_event_mask = np.zeros(len(tokens))
      for entity_mention in entity_mentions:
        mention_id = entity_mention['id']
        entity_id = '-'.join(mention_id.split('-')[:-1])
        cluster_id = cluster_ids[entity_id]
        start = entity_mention['start']
        end = entity_mention['end']
        
        entity = {'doc_id': doc_id,
                  'm_id': mention_id,
                  'sentence_id': sent_id,
                  'entity_type': entity_mention['entity_type'],
                  'tokens_ids': list(range(sen_start+start, sen_start+end)),
                  'tokens': entity_mention['text'],
                  'cluster_id': cluster_id,
                  'singleton': False}
        # entity = get_entity_feature(entity, tokens, dep_head, dep_role, sen_start=sen_start) # TODO
        entity_event_mask[start:end] = 1.
        entities.append(entity)

      for event_mention in event_mentions:
        mention_id = event_mention['id']
        event_id = '-'.join(mention_id.split('-')[:-1])
        cluster_id = cluster_ids[event_id]
        start = event_mention['trigger']['start']
        end = event_mention['trigger']['end']
        
        event = {'doc_id': doc_id,
                 'm_id': mention_id,
                 'arguments': event_mention['arguments'],
                 'sentence_id': sent_id,
                 'event_type': event_mention['event_type'],
                 'tokens_ids': list(range(sen_start+start, sen_start+end)),
                 'tokens': event_mention['trigger']['text'],
                 'cluster_id': cluster_id,
                 'singleton': False}
        entity_event_mask[start:end] = 1.
        events.append(event)      

      # sent = [[sent_id, sen_start+t_idx, t, int(entity_event_mask[t_idx]) > 0, dep_role[t_idx], sen_start+dep_head[t_idx]] for t_idx, t in enumerate(tokens)]
      sent = [[sent_id, sen_start+t_idx, t, int(entity_event_mask[t_idx]) > 0, pos_tags[t_idx]] for t_idx, t in enumerate(tokens)]

      documents[doc_id].extend(sent)
      sen_start += len(tokens)
  
  json.dump(documents, codecs.open(out_prefix+'.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(events, codecs.open(out_prefix+'_events.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(entities, codecs.open(out_prefix+'_entities.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(entities+events, codecs.open(out_prefix+'_mixed.json', 'w', 'utf-8'), indent=4, sort_keys=True)

def get_event_info(data_json, out_prefix):
  '''
  :param data_json: str of filename of the meta info file storing a json list of dicts with keys:
      doc_id: str,
      sent_id: str,
      tokens: list of str,
      entity_mentions: list of dicts of
        id: str,
        text: str,
        start: int, start token idx 
        end: int, end token idx
      event_mentions: list of dicts of
        id: str,
        text: str,
        start: int, start token idx
        end: int, end token idx

  :param out_prefix: str of the prefix of three files:
      1. [prefix].json: dict of 
        [doc_id]: list of [sent id, token id, token, is entity/event]
      2,3. [prefix]_[entities/events].json: store list of dicts of:
      {'doc_id': str, file prefix of the image for the caption,
       'subtopic': '0',
       'm_id': '0',
       'sentence_id': str, order of the sentence,
       'tokens_ids': list of ints, one-indexed position of the tokens of the current mention in the sentence,
       'tokens': str, tokens concatenated with space,
       'cluster_id': int, 
       'tags': '',
       'lemmas': '',
       'singleton': boolean, whether the mention is a singleton} 
  '''
  documents = {}
  events = []
  cluster_ids = {'###SINGLETON###': 0}

  sen_start = 0
  with open(data_json, 'r') as f:
    i = 0
    for line in f:
      # if i > 20:
      #   break
      i += 1
      inst = json.loads(line)
      doc_id = inst['doc_id']
      sent_id = inst['sent_id']
      tokens = inst['tokens']
      entity_mentions = inst['entity_mentions']
      event_mentions = inst['event_mentions']

      if not doc_id in documents:
        documents[doc_id] = []
        sen_start = 0

      entities = {}
      for entity_mention in entity_mentions:
        mention_id = entity_mention['id']
        entity_id = '-'.join(mention_id.split('-')[:-1])
        if not entity_id in cluster_ids:
          cluster_ids[entity_id] = len(cluster_ids)

        cluster_id = cluster_ids[entity_id]
        start = entity_mention['start']
        end = entity_mention['end']
        entity_type = entity_mention['entity_type']
        entities[mention_id] = {
                  'doc_id': doc_id,
                  'sentence_id': sent_id,
                  'm_id': entity_id,
                  'tokens_ids': list(range(sen_start+start, sen_start+end)),
                  'tokens': entity_mention['text'],
                  'cluster_id': cluster_id,
                  'entity_type': entity_type, 
                  'singleton': False
                  }

      for event_mention in event_mentions:
        mention_id = event_mention['id']
        event_id = '-'.join(mention_id.split('-')[:-1])
        if not event_id in cluster_ids:
          cluster_ids[event_id] = len(cluster_ids) 
        cluster_id = cluster_ids[event_id]
        start = event_mention['trigger']['start']
        end = event_mention['trigger']['end']
        event_type = event_mention['event_type']
        arguments = event_mention['arguments']
        
        argument_dicts = []
        for a in arguments:
          arg_dict = entities[a['entity_id']] 
          arg_dict['role'] = a['role']
          argument_dicts.append(arg_dict)

        event = {'doc_id': doc_id,
                 'sentence_id': sent_id,
                 'm_id': mention_id,
                 'tokens_ids': list(range(sen_start+start, sen_start+end)),
                 'tokens': event_mention['trigger']['text'],
                 'cluster_id': cluster_id,
                 'event_type': event_type,
                 'singleton': False,
                 'arguments': argument_dicts 
                }
        events.append(event)
      sen_start += len(tokens)
  json.dump(events, codecs.open(out_prefix+'_events_with_arguments.json', 'w', 'utf-8'), indent=4, sort_keys=True)

def get_entity_features(entity, 
                        tokens,
                        dep_head, dep_role, 
                        sen_start,
                        feature_list=['nsubj', 'pobj', 'amod', 'root']):
  features = {k: [] for k in feature_list}

  tokens_ids = entity['tokens_ids']
  for i, (h, r) in enumerate(zip(dep_head, dep_role)):
    if sen_start+i in tokens_ids:
      continue
    
    if r == 'nsubj' and 'nsubj' in feature_list:
      features['nsubj'].append(tokens[i])

    if r == 'pobj' and 'pobj' in feature_list:
      features['pobj'].append(tokens[i])

    if h in tokens_ids and r == 'amod' and 'amod' in feature_list:
      features['amod'].append(tokens[i])
    
    if r == 'root' and 'root' in feature_list:
      features['head'].append(tokens[i])

  entity['features'] = features
  return entity

def extract_synthetic_visual_event_embeddings(data_dir, out_prefix):
  '''
  :returns event_feats: an Npz object of arrays of shape (num. of events, image embed dim.)
  :returns argument_feats: an Npz object of arrays of shape (num. of events, max. num. of roles per event, image embed dim.)
  :returns visual_event_mentions: a list of dicts of format
      {'doc_id': str,
       'm_id': str,
       'sent_id': str,
       'tokens': str, assumed to be event type 
       'event_type': str,
       'cluster_id': int,
       'arguments': list of dicts of format
                    {'doc_id': str,
                     'm_id': str,
                     'tokens': int, assumed to be cluster id
                     'entity_type': str,
                     'role': str,
                     'cluster_id': int 
                     }
       'singleton': false}
  '''
  event_mentions_train = json.load(codecs.open(os.path.join(data_dir, f'train_events.json')))
  event_mentions_test = json.load(codecs.open(os.path.join(data_dir, f'test_events.json'))) 

  label_dict = {}
  event_stoi = {}
  entity_stoi = {} 
  for m in event_mentions_train + event_mentions_test:
    # Create a visual mention dict for the current document
    if not m['event_type'] in event_stoi:
      event_stoi[m['event_type']] = len(event_stoi)

    if not m['doc_id'] in label_dict:
      label_dict[m['doc_id']] = {}

    m_id = '-'.join(m['m_id'].split('-')[:-1])
    if not m_id in label_dict[m['doc_id']]:
      label_dict[m['doc_id']][m_id] = {'tokens': m['event_type'],
                                       'event_type': m['event_type'],
                                       'cluster_id': m['cluster_id'],
                                       'arguments': {}}

    # Check if the current cluster id is new, and if so, create a new dict within the current event dict; copy the information of the current event to the dict
    for a in m['arguments']:
      a_id = '-'.join(a['m_id'].split('-')[:-1])
      if not a_id in entity_stoi:
        entity_stoi[a_id] = len(entity_stoi)

      if not a_id in label_dict['arguments']:
        label_dict[m['doc_id']][m_id]['arguments'][a_id] = {'tokens': a['cluster_id'],
                                                            'entity_type': a['entity_type'],
                                                            'role': a['role'],
                                                            'cluster_id': a['cluster_id']}
  
  n_event_clusters = len(event_stoi)
  n_entity_clusters = len(entity_stoi)
  print(f'Number of event clusters {n_event_clusters}')
  print(f'Number of entity clusters {n_entity_clusters}')

  visual_mentions = []
  event_feats = {}
  argument_feats = {}
  for doc_idx, doc_id in enumerate(sorted(label_dict)):
    feat_id = f'{doc_id}_{doc_idx}'
    event_feat = []
    argument_feat = []
    for m_id in sorted(label_dict[doc_id]):
      cur_mention = {'doc_id': doc_id,
                     'm_id': m_id,
                     'tokens': label_dict[doc_id][m_id]['tokens'],
                     'event_type': label_dict[doc_id][m_id]['event_type'],
                     'cluster_id': label_dict[doc_id][m_id]['cluster_id'],
                     'arguments': []}
      event_feat.append(event_stoi[cur_mention['tokens']])
      argument_feat.append([])
      for a_id in sorted(label_dict[doc_id]['arguments']):
        a = label_dict[doc_id]['arguments'][a_id]
        cur_mention['arguments'].append({'tokens': a['cluster_id'],
                                         'entity_type': a['entity_type'],
                                         'role': a['role'],
                                         'cluster_id': a['cluster_id']})
        argument_feat[-1].append(cur_mention['tokens'])     
      argument_feat[-1] = to_fixed_length(to_one_hot(argument_feat[-1], n_entity_clusters), pad_val=-1)
      visual_mentions.append(cur_mention) 
      
    # Extract one-hot vectors
    event_feats[feat_id] = to_one_hot(event_feat, n_event_clusters) 
    argument_feats[feat_id] = np.asarray(argument_feat) 

  json.dump(visual_mentions, open(out_prefix+'_visual.json', 'w'), indent=2)
  np.savez(out_prefix+'_event_features.npz', **event_feats)
  np.savez(out_prefix+'_entity_features.npz', **argument_feats)  

def to_one_hot(sent, K):
  sent = np.asarray(sent)
  if len(sent.shape) < 2:
    es = np.eye(K)
    sent = np.asarray([es[int(w)] if w < K else 1./K*np.ones(K) for w in sent])
    return sent
  else:
    return sent

def to_fix_length(x, L, pad_val=-1):
  shape = x.shape[1:]
  if x.shape[0] < L:
    pad = [pad_val*np.ones(shape)[np.newaxis] for _ in range(L-x.shape[0])]
    x = np.concatenate([x]+pad)
  else:
    x = x[:L]
  return x


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--task', type=int)
  args = parser.parse_args()

  data_dir = 'ace/'
  if args.task == 0:
    if not os.path.exists(os.path.join(data_dir, 'mentions')):
      os.mkdir(os.path.join(data_dir, 'mentions'))
    for split in ['train', 'dev', 'test']:
      data_json = os.path.join(data_dir, '{}.oneie.json'.format(split))
      out_prefix = os.path.join(data_dir, 'mentions', split)
      get_mention_doc(data_json, out_prefix)
      get_event_info(data_json, out_prefix)
  elif args.task == 1:
    out_prefix = os.path.join(data_dir, 'mentions', 'traintest')
    extract_synthetic_visual_event_embeddings(data_dir, out_prefix)
