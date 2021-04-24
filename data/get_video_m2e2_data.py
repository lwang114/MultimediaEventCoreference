# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import os
import collections
import nltk
from nltk.stem import WordNetLemmatizer
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
np.random.seed(2)

PUNCT = [',', '.', '\'', '\"', ':', ';', '?', '!', '<', '>', '~', '%', '$', '|', '/', '@', '#', '^', '*']
def get_mention_doc_oneie(data_dir, out_prefix):
  '''
  Deal with ASR transcripts and OneIE results.
  
  :param data_dir: str of directory of text files with each line in each file containing a dict of format
      doc_id: str,
      sent_id: [doc_id]-[sentence idx],
      tokens: a list of strs, tokenized sentence,
      graph: a dict of format
          {'trigger': a list of list of [start idx, end idx+1, event type, score],
           'entities': a list of list of [start idx, end idx+1, entity type, score],
           'roles': a list of list of [event idx, entity idx, role, score]}
  '''
  lemmatizer = WordNetLemmatizer()
  data_files = os.listdir(data_dir)
  documents = {}
  mentions = []
  for data_file in data_files: # XXX
    print(data_file)
    doc_id = None
    tokens = []
    cur_mentions = []
    start_sent = 0
    with open(os.path.join(data_dir, data_file), 'r') as f:
      for line in f:
        sent_dict = json.loads(line)
        graph = sent_dict['graph']
        triggers_info = graph['triggers']
        entities_info = graph['entities']
        roles_info = graph['roles']

        if not doc_id:
          doc_id = sent_dict['doc_id']
        sent_idx = int(sent_dict['sent_id'].split('-')[-1])
        postags = [t[1] for t in nltk.pos_tag(sent_dict['tokens'], tagset='universal')]
        tokens.extend([[sent_idx, start_sent+i, token, tag]\
                      for i, (token, tag) in enumerate(zip(sent_dict['tokens'], postags))])
        
        for trigger_idx, trigger in enumerate(triggers_info):
          trigger_tokens = sent_dict['tokens'][trigger[0]:trigger[1]]
          mention = {'doc_id': doc_id,
                     'tokens': ' '.join(trigger_tokens),
                     'tokens_ids': list(range(start_sent+trigger[0], start_sent+trigger[1])),
                     'event_type': trigger[2],
                     'arguments': [],
                     'cluster_id': 0}
          pos_abbrev = postags[trigger[1]-1][0].lower() if postags[trigger[1]-1] in ['NOUN', 'VERB', 'ADJ'] else 'n'   
          head_lemma = lemmatizer.lemmatize(trigger_tokens[-1], pos=pos_abbrev) 
          mention['head_lemma'] = head_lemma
          for role in roles_info:
            if role[0] == trigger_idx:
              entity = entities_info[role[1]]
              mention['arguments'].append({'start': start_sent+entity[0],
                                           'end': start_sent+entity[1]-1,
                                           'token': ' '.join(sent_dict['tokens'][entity[0]:entity[1]]),
                                           'role': role[2]})
          cur_mentions.append(mention)
        start_sent += len(sent_dict['tokens'])

    if len(cur_mentions) > 0:
      mentions.extend(cur_mentions)
      documents[doc_id] = tokens
  
  print(f'Number of documents: {len(documents)}')
  json.dump(documents, open(f'{out_prefix}.json', 'w'), indent=2)
  json.dump(mentions, open(f'{out_prefix}_events.json', 'w'), indent=2)

def get_mention_doc_m2e2(data_json, out_prefix, inclusive=False):
  '''
  :param data_json: str of filename of the meta info file storing a list of dicts with keys:
      sentence_id: str, file prefix of the image for the caption
      words: list of str, tokens of the sentence 
      golden-entity-mentions: list of dicts of 
        {'entity-type': [entity type],
         'text': [tokens concatenated using space],
         'start': int, start of the entity,
         'end': int, end of the entity}
      golden-event-mentions: list of dicts of 
        {'trigger': {'start' int,
                     'end': int,
                     'text': str}
         'event_type': str,
         'arguments': [...]}
      (Optional) coreference:
        {'entities': {[cluster id]: [mention idx, mention tag id], ...},
         'events': {[cluster id]: [mention idx, mention tag id], ...}
        }
      inclusive: boolean, true if the token interval is [start, end] and false if it is [start, end)
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
         'tags': '',
         'lemmas': '',
         'cluster_id': int,
         'cluster_desc': '',
         'singleton': boolean, whether the mention is a singleton} 
  '''
  sen_dicts = json.load(open(data_json))
  outs = {}
  event2coref = {}
  entity2coref = {}
  event_cluster2id = {}
  entity_cluster2id = {}
  entities = []
  events = []
  n_event_cluster = 0
  n_event_corefs = 0
  n_entity_cluster = 0
  n_entity_corefs = 0
  sen_start = 0
  end_inc = 1 if inclusive else 0 
  cur_id = ''
  for sen_dict in sen_dicts: # XXX
    doc_id = sen_dict['image']
    sent_id = sen_dict['sentence_id']
    tokens = sen_dict['words']
    entity_mentions = sen_dict['golden-entity-mentions']
    event_mentions = sen_dict['golden-event-mentions']

    if doc_id != cur_id:
      cur_id = doc_id
      sen_start = 0
      if len(event2coref) > 0:
        coref2event = collections.defaultdict(list)
        coref2entity = collections.defaultdict(list)
        for e, c in event2coref.items():
          coref2event[c].append(e)
        for e, c in entity2coref.items():
          coref2entity[c].append(e)

      event2coref = {} # Mapping from event mention id to its integer cluster idx
      entity2coref = {} # Mapping from entity mention id to its integer cluster idx
      event_cluster2id = {} # Mapping from event cluster id to integer cluster idx
      entity_cluster2id = {} # Mapping from entity cluster id to integer cluster idx

    if 'coreference' in sen_dict:
      coreference = sen_dict['coreference']
      entity_mention_ids = sen_dict['mention_ids']['entities']
      event_mention_ids = sen_dict['mention_ids']['events']
      # Create coreference mapping
      for cluster_id in sorted(coreference['events']):
        if not cluster_id in event_cluster2id:
          n_event_cluster += 1
          event_cluster2id[cluster_id] = n_entity_cluster + n_event_cluster 

        for mention in coreference['events'][cluster_id]:
          event2coref[mention[1]] = event_cluster2id[cluster_id]

      for mention_id in event_mention_ids:
        if not mention_id in event2coref:
          n_event_cluster += 1
          event2coref[mention_id] = n_entity_cluster + n_event_cluster


      for cluster_id in sorted(coreference['entities']):
        if not cluster_id in entity_cluster2id:
          n_entity_cluster += 1
          entity_cluster2id[cluster_id] = n_entity_cluster + n_event_cluster

        for mention in coreference['entities'][cluster_id]:
          entity2coref[mention[1]] = entity_cluster2id[cluster_id]
      
      for mention_id in entity_mention_ids:
        if not mention_id in entity2coref:
          n_entity_cluster += 1
          entity2coref[mention_id] = n_entity_cluster + n_event_cluster 
      
    coref2event = collections.defaultdict(list)
    coref2entity = collections.defaultdict(list)
    for e, c in event2coref.items():
      coref2event[c].append(e)
    for e, c in entity2coref.items():
      coref2entity[c].append(e)
    
    entity_mask = [0]*len(tokens)
    event_mask = [0]*len(tokens)
    # Create dict for [out_prefix]_entities.json
    for m_idx, mention in enumerate(entity_mentions):
        for pos in range(mention['start'], mention['end']+end_inc):
          entity_mask[pos] = 1
        
        if 'coreference' in sen_dict:
          m_id = entity_mention_ids[m_idx] 
          cluster_id = entity2coref[m_id]
        else:  
          cluster_id = '0'
        entities.append({'doc_id': doc_id,
                         'subtopic': '0',
                         'm_id': '0',
                         'sentence_id': sent_id,
                         'tokens_ids': list(range(sen_start+mention['start'], sen_start+mention['end']+end_inc)),
                         'tokens': ' '.join(tokens[mention['start']:mention['end']+end_inc]),
                         'entity_type': mention['entity_type'], 
                         'tags': '',
                         'lemmas': '',
                         'cluster_id': cluster_id,
                         'cluster_desc': '',
                         'singleton': False})

    # Create dict for [out_prefix]_events.json
    for m_idx, mention in enumerate(event_mentions): 
        try: # XXX
          start = mention['trigger']['start']
          end = mention['trigger']['end']
        except:
          start = mention['start']
          end = mention['end']

        for pos in range(start, end+end_inc):
          event_mask[pos] = 1

        arguments = []
        for arg_info in mention['arguments']:
          start = arg_info['start']
          end = arg_info['end']
          arguments.append((start, end))
          
        if 'coreference' in sen_dict:
          m_id = event_mention_ids[m_idx]
          cluster_id = event2coref[m_id]
        else:
          cluster_id = '0'
        events.append({'doc_id': doc_id,
                       'subtopic': '0',
                       'm_id': '0',
                       'sentence_id': sent_id,
                       'tokens_ids': list(range(sen_start+start, sen_start+end+end_inc)),
                       'tokens': ' '.join(tokens[start:end+end_inc]),
                       'event_type': mention['event_type'], 
                       'arguments': arguments,
                       'tags': '',
                       'lemmas': '',
                       'cluster_id': cluster_id,
                       'cluster_desc': '',
                       'singleton': False})

    # Create dict for [out_prefix].json
    if not doc_id in outs: 
      outs[doc_id] = []
    
    for idx, token in enumerate(tokens):
      outs[doc_id].append([sent_id, sen_start+idx, token, entity_mask[idx] > 0 or event_mask[idx] > 0])
    sen_start += len(tokens)
      
  json.dump(outs, codecs.open(out_prefix+'.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(entities, codecs.open(out_prefix+'_entities.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(events, codecs.open(out_prefix+'_events.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(entities+events, codecs.open(out_prefix+'_mixed.json', 'w', 'utf-8'), indent=4, sort_keys=True)

def extract_visual_event_embeddings(data_dir, csv_dir, mapping_file, duration_file, annotation_file, out_prefix='mmaction_event_feats', k=5):
  ''' Extract visual event embeddings
  :param config: str, config file for the dataset,
  :param split: str, 'train' or 'test',
  :param in_feat_file: str, frame-level video feature file,
  :param mapping_file: str, filename of a mapping from doc id to video id
  :param duration_file: str, filename of duration info of the videos of the format
      {
        'duration_second': float,
        }
  :param annotation_file: str, filename of the visual event annotations of the format
      {
        [description_id]: list of mapping of 
            {'Temporal_Boundary': [float, float],    
             'Event_Type': int,
             'Key_Frames': [list of mapping of 
                 {'Timestamp': float,
                  'Arguments': [{'Bounding_Box': [x_min, y_min, x_max, y_max],
                                 'ROLE_TYPE': int,
                                 'Entity_Type': int,
                                 'Event_Coreference: null or int'}]}]}
        } 
  :return out_prefix: str
  '''
  mapping_dict = json.load(open(mapping_file))
  dur_dict = json.load(open(duration_file))
  ann_dict = json.load(open(annotation_file))
  id2desc = {v['id'].split('v=')[-1]:k for k, v in mapping_dict.items()}
  doc_ids = sorted(id2desc)
  is_csv_used = {fn:0 for fn in os.listdir(csv_dir)}

  event_feats = {}
  event_labels = {}
  argument_feats = {}
  argument_labels = {}
  event_frequency = {}
  entity_frequency = {}
  for idx, doc_id in enumerate(doc_ids): # XXX
    # Convert the .csv file to numpy array
    desc = id2desc[doc_id]
    for punct in PUNCT:
      desc = desc.replace(punct, '')
    csv_file = os.path.join(csv_dir, desc+'.csv')
    if not os.path.exists(csv_file) or not desc+'.mp4' in ann_dict:
      print('Skip description id: {}'.format(desc))
      continue
    is_csv_used[desc+'.csv'] = 1
    
    # Extract frame-level features
    frame_feats = []
    skip_header = 1
    for line in codecs.open(csv_file, 'r', 'utf-8'):
      if skip_header:
        skip_header = 0
        continue
      segments = line.strip().split(',')
      if len(segments) == 0:
        print('Empty line')
        break
      frame_feats.append([float(x) for x in segments[-400:]])
    frame_feats = np.asarray(frame_feats)
    nframes = frame_feats.shape[0] 

    # Extract event features
    event_label = []
    event_feat = []
    argument_label = []
    argument_feat = [] 

    dur = dur_dict[desc]['duration_second']
    for event_dict in ann_dict[desc+'.mp4']:
      event_type = event_dict['Event_Type']
      event_label.append(event_type)
      if not event_type in event_frequency:
        event_frequency[event_type] = 1
      else:
        event_frequency[event_type] += 1 
      start_time, end_time = event_dict['Temporal_Boundary']
      start, end = int(start_time / dur * nframes), int(end_time / dur * nframes)
      
      cur_arg_feat = []
      cur_arg_label = []
      for frame_dict in event_dict['Key_Frames']:
        timestamp = frame_dict['Timestamp']
        frame_idx = int(timestamp / dur * nframes)
        for arg_dict in frame_dict['Arguments']:
          entity_type = arg_dict.get('Entity_Type', -1)
          if not entity_type in entity_frequency:
            entity_frequency[entity_type] = 1
          else:
            entity_frequency[entity_type] += 1

          cur_arg_feat.append(frame_feats[frame_idx])
          cur_arg_label.append([entity_type, arg_dict.get('ROLE_TYPE', -1)])

      cur_arg_feat = np.stack(cur_arg_feat)
      cur_arg_label = np.asarray(cur_arg_label)
      cur_arg_feat = to_fix_length(cur_arg_feat, L=5, pad_val=-1)
      cur_arg_label = to_fix_length(cur_arg_label, L=5, pad_val=-1)
      argument_feat.append(cur_arg_feat)
      argument_label.append(cur_arg_label)

      # Extract the top k subspace vectors
      '''
      if end - start + 1 > k:
        event_feat.append(PCA(n_components=k).fit(frame_feats[start:end+1]).components_)
      elif end - start:
        feat = PCA(n_components=end-start+1).fit(frame_feats[start:end+1]).components_
        feat = np.concatenate([feat, np.zeros((k-(end-start+1), feat.shape[-1]))], axis=0)
        event_feat.append(feat)
      else:
        event_feat.append(np.tile(frame_feats[start][np.newaxis], (k, 1)))
      '''
      event_feat.append(frame_feats[start:end+1].mean(axis=0))
    
    event_feat = np.stack(event_feat, axis=0)
    argument_feat = np.stack(argument_feat, axis=0)
    argument_label = np.stack(argument_label, axis=0)

    feat_id = '{}_{}'.format(doc_id, idx)
    print(feat_id, event_feat.shape)
    
    event_labels[feat_id] = np.asarray(event_label)
    argument_labels[feat_id] = argument_label
    event_feats[feat_id] = event_feat
    argument_feats[feat_id] = argument_feat
  np.savez(f'{out_prefix}.npz', **event_feats)
  np.savez(f'{out_prefix}_argument_feat.npz', **argument_feats)

  # Convert the labels to one-hot
  n_event_types = max(int(t) for t in event_frequency) + 1
  event_labels_onehot = {k:np.eye(n_event_types)[l] for k, l in event_labels.items()} 
  np.savez(f'{out_prefix}_labels.npz', **event_labels_onehot)
  np.savez(f'{out_prefix}_argument_labels.npz', **argument_labels)
  json.dump(entity_frequency, open(f'{out_prefix}_entity_frequency.npz', 'w'), indent=4, sort_keys=True)
  json.dump(event_frequency, open(f'{out_prefix}_event_frequency.json', 'w'), indent=4, sort_keys=True)
 
def extract_visual_embeddings(csv_dirs, out_prefix, mapping_file=None, image_ids=None):
  # Create a mapping from Youtube id to short description
  id2desc = None
  if mapping_file:
    mapping_dict = json.load(open(mapping_file))
    id2desc = {v['id'].split('v=')[-1]:k for k, v in mapping_dict.items()}
    doc_ids = sorted(id2desc)
  else:
    doc_ids = [os.path.join(csv_dir, '.'.join(fn.split('.')[:-1])) for csv_dir in csv_dirs for fn in os.listdir(csv_dir)] 

  img_feats = {}
  for idx, doc_id in enumerate(doc_ids): # XXX
    print(idx, doc_id)
    # Convert the .csv file to numpy array 
    if id2desc is not None:
      desc = id2desc[doc_id]
      for punct in PUNCT:
        desc = desc.replace(punct, '')
      csv_file = os.path.join(csv_dirs[0], desc+'.csv')
    else:
      csv_file = os.path.join(doc_id+'.csv')
      doc_id = doc_id.split('/')[-1].replace(' ', '_') # Replace space with '_' to be consistent with OneIE 

    if not os.path.exists(csv_file):
      print('File {} not found'.format(csv_file))
      continue

    img_feat = []
    skip_header = 1
    for line in codecs.open(csv_file, 'r', 'utf-8'):
      if skip_header:
        skip_header = 0
        continue
      segments = line.strip().split(',') 
      if len(segments) == 0:
        print('Empty line')
        break
      img_feat.append([float(x) for x in segments[-400:]])
    img_feat = np.asarray(img_feat)
    img_feats['{}_{}'.format(doc_id, idx)] = img_feat
  np.savez(out_prefix, **img_feats)

def train_test_split(feat_file, test_id_file, mapping_file, out_prefix): # TODO random split
  mapping_dict = json.load(open(mapping_file)) 
  id2desc = {v['id'].split('v=')[-1]:k for k, v in mapping_dict.items()}
  feats = np.load(feat_file)
  feat_ids = sorted(feats, key=lambda x:int(x.split('_')[-1])) 
  test_id_dict = json.load(open(test_id_file))
  train_feats = {}
  test_feats = {}
  
  for k in feat_ids:
    doc_id = '_'.join(k.split('_')[:-1])
    if doc_id in test_id_dict:
      test_feats[k] = feats[k] 
    else:
      train_feats[k] = feats[k]

  np.savez('{}_train.npz'.format(out_prefix), **train_feats)
  np.savez('{}_test.npz'.format(out_prefix), **test_feats)

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

  data_dir = 'video_m2e2/mentions/'
  mapping_file = 'video_m2e2/video_m2e2.json'
  test_desc_file = 'video_m2e2/unannotatedVideos_textEventCount.json'
  data_json = 'video_m2e2/grounding_video_m2e2.json'
  csv_dir = os.path.join(data_dir, 'mmaction_feat')
  
  if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
    os.mkdir(os.path.join(data_dir, 'mentions'))
    os.mkdir(os.path.join(data_dir, 'gold'))

  if args.task == 0:  
    out_prefix = os.path.join(data_dir, 'train')
    get_mention_doc(data_json, out_prefix, inclusive=False) 
    data_json = 'video_m2e2/grounding_video_m2e2_test.json'
    out_prefix = os.path.join(data_dir, 'test')
    get_mention_doc_m2e2(data_json, out_prefix, inclusive=True)
  elif args.task == 1:
    save_gold_conll_files(out_prefix+'.json', out_prefix+'_mixed.json', os.path.join(data_dir, 'gold')) 
  elif args.task == 2:
    data_dir = 'video_m2e2'
    csv_dir = os.path.join(data_dir, 'mmaction_feat')
    out_prefix = '{}/{}'.format(data_dir, csv_dir.split('/')[-1])
    extract_visual_embeddings([csv_dir], out_prefix, mapping_file=mapping_file)
  elif args.task == 3:
    train_test_split('{}/{}.npz'.format(data_dir, csv_dir.split('/')[-1]), os.path.join(data_dir, 'test.json'), mapping_file, out_prefix)
  elif args.task == 4:
    # Xudong's split
    out_dir = 'video_m2e2_old/mentions/'
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    documents = json.load(open(os.path.join(data_dir, 'train.json')))
    entity_mentions = json.load(open(os.path.join(data_dir, 'train_entities.json')))
    event_mentions = json.load(open(os.path.join(data_dir, 'train_events.json')))
    mixed_mentions = json.load(open(os.path.join(data_dir, 'train_mixed.json')))
    anno_train = json.load(open(os.path.join(data_dir, '../anet_anno_train.json')))
    mapping_dict = json.load(open(os.path.join(data_dir, '../video_m2e2.json')))
    id2desc = dict()
    for k, v in mapping_dict.items():
      for punct in PUNCT:
        k = k.replace(punct, '')
      id2desc[v['id'].split('v=')[-1]] = k
    
    n_docs = len(documents)
    doc_ids = sorted(documents)
    train_ids = [doc_id for doc_id in doc_ids if id2desc[doc_id] in anno_train]
    # n_train = int(0.75 * n_docs)
    # train_idxs = np.random.permutation(n_docs)[:n_train]
    # train_ids = [doc_ids[i] for i in train_idxs] 
    test_ids = [doc_id for doc_id in doc_ids if not doc_id in train_ids]
    
    documents_train = {}
    mixed_mentions_train = [m for m in mixed_mentions if m['doc_id'] in train_ids]
    entity_mentions_train = [m for m in entity_mentions if m['doc_id'] in train_ids]
    event_mentions_train = [m for m in event_mentions if m['doc_id'] in train_ids]
    for train_id in train_ids:
      documents_train[train_id] = deepcopy(documents[train_id])

    documents_test = {}
    mixed_mentions_test = [m for m in mixed_mentions if m['doc_id'] in test_ids]
    entity_mentions_test = [m for m in entity_mentions if m['doc_id'] in test_ids]
    event_mentions_test = [m for m in event_mentions if m['doc_id'] in test_ids]
    for test_id in test_ids:
      documents_test[test_id] = deepcopy(documents[test_id])

    json.dump(documents_train, open(os.path.join(out_dir, 'train.json'), 'w'), indent=2)
    json.dump(mixed_mentions_train, open(os.path.join(out_dir, 'train_mixed.json'), 'w'), indent=2)
    json.dump(entity_mentions_train, open(os.path.join(out_dir, 'train_entities.json'), 'w'), indent=2)
    json.dump(event_mentions_train, open(os.path.join(out_dir, 'train_events.json'), 'w'), indent=2)
    
    json.dump(documents_test, open(os.path.join(out_dir, 'test.json'), 'w'), indent=2)
    json.dump(mixed_mentions_test, open(os.path.join(out_dir, 'test_mixed.json'), 'w'), indent=2)
    json.dump(entity_mentions_test, open(os.path.join(out_dir, 'test_entities.json'), 'w'), indent=2)
    json.dump(event_mentions_test, open(os.path.join(out_dir, 'test_events.json'), 'w'), indent=2)
  elif args.task == 5:
    annotation_file = os.path.join(data_dir, '../master.json')
    duration_file = os.path.join(data_dir, '../anet_anno.json')
    if not os.path.exists(duration_file):
      train_duration_file = os.path.join(data_dir, '../anet_anno_train.json')
      test_duration_file = os.path.join(data_dir, '../anet_anno_val.json')
      train_dur_dict = json.load(open(train_duration_file))
      test_dur_dict = json.load(open(test_duration_file))
      train_dur_dict.update(test_dur_dict)
      json.dump(train_dur_dict, open(duration_file, 'w'), indent=2)
    out_prefix = os.path.join(data_dir, 'train_mmaction_event_feat')
    extract_visual_event_embeddings(data_dir, csv_dir, mapping_file, duration_file, annotation_file, out_prefix=out_prefix)
  elif args.task == 6:
    out_prefix = os.path.join(data_dir, 'train_mmaction_event_feat')
    embed_file = f'{out_prefix}.npz'
    label_file = f'{out_prefix}_labels.npz'
    freq_file = f'{out_prefix}_event_frequency.json'
    ontology_file = os.path.join(data_dir, '../ontology.json')
    visualize_features(embed_file, label_file, ontology_file, freq_file, out_prefix=f'{out_prefix}_tsne')
  elif args.task == 7:
    anno_dir = 'video_m2e2/json/'
    out_prefix = os.path.join(data_dir, 'train_unlabeled')
    get_mention_doc_oneie(anno_dir, out_prefix)
  elif args.task == 8:
    anno_dir = 'video_m2e2/json_asr/'
    out_prefix = os.path.join(data_dir, 'train_asr')
    get_mention_doc_oneie(anno_dir, out_prefix)
  elif args.task == 9: # Unlabeled video feature extraction
    data_dir = 'video_m2e2'
    csv_dir = os.path.join(data_dir, 'output_feat')
    out_prefix = os.path.join(data_dir, 'mentions/train_unlabeled_mmaction_feat')
    extract_visual_embeddings([csv_dir], out_prefix=out_prefix)
  elif args.task == 10:
    data_dir = 'video_m2e2'
    csv_dirs = [os.path.join(data_dir, 'mmaction_feat'), os.path.join(data_dir, 'output_feat')]
    out_prefix = os.path.join(data_dir, 'mentions/train_asr_mmaction_feat')
    extract_visual_embeddings(csv_dirs, out_prefix=out_prefix)
