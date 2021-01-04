# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import os
import collections
from conll import write_output_file

PUNCT = [',', '.', '\'', '\"', ':', ';', '?', '!', '<', '>', '~', '%', '$', '|', '/', '@', '#', '^', '*']
def get_mention_doc(data_json, out_prefix, inclusive=False):
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
  for sen_dict in sen_dicts:
    doc_id = sen_dict['image']
    # XXX '0_{}'.format(sen_dict['image']) # Prepend a dummy topic id to run on coref
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
          event_cluster2id[cluster_id] = n_event_cluster 
          n_event_cluster += 1

        for mention in coreference['events'][cluster_id]:
          event2coref[mention[1]] = event_cluster2id[cluster_id]
          print('doc id: {}, mention id: {}, event cluster id: {}, event cluster idx: {}'.format(doc_id, mention[1], cluster_id, event2coref[mention[1]])) # XXX

      for mention_id in event_mention_ids:
        if not mention_id in event2coref:
          event2coref[mention_id] = n_event_cluster
          n_event_cluster += 1

      for cluster_id in sorted(coreference['entities']):
        if not cluster_id in entity_cluster2id:
          entity_cluster2id[cluster_id] = n_entity_cluster
          n_entity_cluster += 1

        for mention in coreference['entities'][cluster_id]:
          entity2coref[mention[1]] = entity_cluster2id[cluster_id]
          print('doc id: {}, mention id: {}, entity cluster id: {}, entity cluster idx: {}'.format(doc_id, mention[1], cluster_id, entity2coref[mention[1]])) # XXX
      
      for mention_id in entity_mention_ids:
        if not mention_id in entity2coref:
          entity2coref[mention_id] = n_entity_cluster 
          n_entity_cluster += 1
      
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

def save_gold_conll_files(doc_json, mention_json, dir_path):
  documents = json.load(open(doc_json))
  mentions = json.load(open(mention_json))

  # Extract mention dicts
  label_dict = collections.defaultdict(dict)
  for m in mentions:
    if len(m['tokens_ids']) == 0:
      label_dict[m['doc_id']][(-1, -1)] = m['cluster_id']
    else:
      start = min(m['tokens_ids'])
      end = max(m['tokens_ids'])
      label_dict[m['doc_id']][(start, end)] = m['cluster_id']
  
  doc_ids = sorted(documents)
  for doc_id in doc_ids:
    document = documents[doc_id]
    cur_label_dict = label_dict[doc_id]
    start_ends = np.asarray([[start, end] for start, end in sorted(cur_label_dict)])
    starts = start_ends[:, 0]
    ends = start_ends[:, 1]

    # Extract clusters
    clusters = collections.defaultdict(list)
    for m_idx, span in enumerate(sorted(cur_label_dict)):
      cluster_id = cur_label_dict[span]
      clusters[cluster_id].append(m_idx)
    non_singletons = {}
    non_singletons = {cluster: ms for cluster, ms in clusters.items() if len(ms) > 1}
    doc_name = doc_id
    write_output_file({doc_id:document}, non_singletons, [doc_id]*len(cur_label_dict), starts, ends, dir_path, doc_name)

def extract_image_embeddings(data_dir, csv_dir, mapping_file, out_prefix, image_ids=None):
  # Create a mapping from Youtube id to short description
  mapping_dict = json.load(open(mapping_file))
  id2desc = {v['id'].split('v=')[-1]:k for k, v in mapping_dict.items()}
  doc_ids = sorted(id2desc)
  is_csv_used = {fn:0 for fn in os.listdir(csv_dir)}

  img_feats = {}
  for idx, doc_id in enumerate(doc_ids): # XXX
    print(idx, doc_id)
    # Convert the .csv file to numpy array 
    desc = id2desc[doc_id]
    for punct in PUNCT:
      desc = desc.replace(punct, '')
    csv_file = os.path.join(csv_dir, desc+'.csv')
    if not os.path.exists(csv_file):
      print('File {} not found'.format(csv_file))
      continue
    is_csv_used[desc+'.csv'] = 1 

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

  json.dump(is_csv_used, open('is_csv_used.json', 'w'), indent=4)
  np.savez(out_prefix, **img_feats)

def train_test_split(feat_file, test_id_file, mapping_file, out_prefix):
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

if __name__ == '__main__':
  data_dir = 'data/video_m2e2/mentions/'
  mapping_file = 'm2e2/data/video_m2e2/video_m2e2.json'
  test_desc_file = 'm2e2/data/video_m2e2/unannotatedVideos_textEventCount.json'
  csv_dir = os.path.join(data_dir, 'mmaction_feat')
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    os.mkdir(os.path.join(data_dir, 'mentions'))
    os.mkdir(os.path.join(data_dir, 'gold'))
  data_json = 'm2e2/data/video_m2e2/grounding_video_m2e2.json'
  out_prefix = os.path.join(data_dir, 'train')
  get_mention_doc(data_json, out_prefix, inclusive=False) 
  # data_json = 'm2e2/data/video_m2e2/grounding_video_m2e2_test.json'
  # out_prefix = os.path.join(data_dir, 'test')
  # get_mention_doc(data_json, out_prefix, inclusive=True)
  # save_gold_conll_files(out_prefix+'.json', out_prefix+'_mixed.json', os.path.join(data_dir, 'gold')) 
  out_prefix = '{}/{}'.format(data_dir, csv_dir.split('/')[-1])
  # extract_image_embeddings(data_dir, csv_dir, mapping_file, out_prefix)
  # train_test_split('{}/{}.npz'.format(data_dir, csv_dir.split('/')[-1]), os.path.join(data_dir, 'test.json'), mapping_file, out_prefix)
