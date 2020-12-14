# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import os
import collections
from conll import write_output_file

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
    doc_id = '0_{}'.format(sen_dict['image']) # Prepend a dummy topic id to run on coref
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
    for m_id, mention in zip(entity_mention_ids, entity_mentions):
        for pos in range(mention['start'], mention['end']+end_inc):
          entity_mask[pos] = 1
        
        if 'coreference' in sen_dict:
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
    for m_id, mention in zip(event_mention_ids, event_mentions): 
        try: # XXX
          start = mention['trigger']['start']
          end = mention['trigger']['end']
        except:
          start = mention['start']
          end = mention['end']

        for pos in range(start, end+end_inc):
          event_mask[pos] = 1

        if 'coreference' in sen_dict:
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

def print_stats(entity_mentions, event_mentions, entity_clusters, event_clusters):
    print('Event clusters: {}'.format(len(event_clusters)))
    print('Event mentions: {}'.format(len(event_mentions)))
    print('Event singletons mentions: {}'.format(
        sum([1 for l in event_mentions if l['singleton']])))
    print('Entity clusters: {}'.format(len(entity_clusters)))
    print('Entity mentions: {}'.format(len(entity_mentions)))
    print('Entity singletons mentions: {}'.format(
        sum([1 for l in entity_mentions if l['singleton']])))
  
def get_clusters(mentions):
    clusters = collections.defaultdict(list)
    for i, mention in enumerate(mentions):
        cluster_id = mention['cluster_id']
        # clusters[cluster_id] = [] if cluster_id not in clusters else clusters[cluster_id]
        clusters[cluster_id].append(i)

    return clusters

def save_gold_conll_files(documents, mentions, clusters, dir_path, doc_name):
    non_singletons = {cluster: ms for cluster, ms in clusters.items() if len(ms) > 1}
    doc_ids = [m['doc_id'] for m in mentions]
    starts = [min(m['tokens_ids']) for m in mentions]
    ends = [max(m['tokens_ids']) for m in mentions]
    print('len(starts), len(ends): ', len(starts), len(ends))
    write_output_file(documents, non_singletons, doc_ids, starts, ends, dir_path, doc_name)
  
if __name__ == '__main__':
  data_dir = 'data/video_m2e2'
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    os.mkdir(os.path.join(data_dir, 'mentions'))
    os.mkdir(os.path.join(data_dir, 'gold'))
  data_json = '../m2e2/data/video_m2e2/grounding_video_m2e2_test.json' # XXX
  out_prefix = os.path.join(data_dir, 'mentions/test') # XXX
  get_mention_doc(data_json, out_prefix, inclusive=True) # XXX
  type = 'test'
  mentions_path = 'data/video_m2e2/mentions'
  gold_conll_path = 'data/video_m2e2/gold'
  
  # Save docs and mentions files
  with open(os.path.join(mentions_path, '{}.json'.format(type)), 'r') as f:
    docs = json.load(f)
  with open(os.path.join(mentions_path, '{}_events.json'.format(type)), 'r') as f:
    events = json.load(f)
  with open(os.path.join(mentions_path, '{}_entities.json'.format(type)), 'r') as f:
    entities = json.load(f)
  with open(os.path.join(mentions_path, '{}_mixed.json'.format(type)), 'r') as f:
    mixed = json.load(f)

  event_clusters, entity_clusters = get_clusters(events), get_clusters(entities)
  mixed_clusters = get_clusters(mixed)

  print_stats(entities, events, entity_clusters, event_clusters)

  event_path = os.path.join(gold_conll_path, '{}_events_gold.conll'.format(type))
  entity_path = os.path.join(gold_conll_path, '{}_entities_gold.conll'.format(type))
  mixed_path = os.path.join(gold_conll_path, '{}_mixed_gold.conll'.format(type))

  save_gold_conll_files(docs, events, event_clusters, gold_conll_path, '{}_events'.format(type))
  save_gold_conll_files(docs, entities, entity_clusters, gold_conll_path, '{}_entities'.format(type))
  save_gold_conll_files(docs, mixed, mixed_clusters, gold_conll_path, '{}_mixed'.format(type))
