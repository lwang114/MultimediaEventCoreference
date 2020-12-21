# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import os
import collections

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
    doc_id = '_'.join(sen_dict['image'][0].split('_')[:-1])
    sent_id = sen_dict['sentence_id']
    tokens = sen_dict['words']
    entity_mentions = sen_dict['golden-entity-mentions']
    event_mentions = sen_dict.get('golden-event-mentions', [])
    
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
    
    entity_mask = [0]*len(tokens)
    event_mask = [0]*len(tokens)
    # Create dict for [out_prefix]_entities.json
    for m_id, mention in enumerate(entity_mentions):
        for pos in range(mention['start'], mention['end']+end_inc):
          entity_mask[pos] = 1
        
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
    for m_id, mention in enumerate(event_mentions): 
        try: # XXX
          start = mention['trigger']['start']
          end = mention['trigger']['end']
        except:
          start = mention['start']
          end = mention['end']

        for pos in range(start, end+end_inc if not inclusive else end):
          event_mask[pos] = 1

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
  
if __name__ == '__main__':
  data_dir = 'data/m2e2'
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    os.mkdir(os.path.join(data_dir, 'mentions'))
    os.mkdir(os.path.join(data_dir, 'gold'))
  data_json = 'm2e2/data/m2e2_rawdata/article_event.json' # XXX
  out_prefix = os.path.join(data_dir, 'mentions/test') # XXX
  get_mention_doc(data_json, out_prefix, inclusive=True)
