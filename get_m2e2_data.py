# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import os

def get_mention_doc(data_json, out_prefix):
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
  entities = []
  events = []
  for sen_dict in sen_dicts:
    doc_id = sen_dict['image'] 
    sent_id = sen_dict['sentence_id'].split('-')[-1]
    tokens = sen_dict['words']
    entity_mentions = sen_dict['golden-entity-mentions']    
    event_mentions = sen_dict['golden-event-mentions']

    entity_mask = [0]*len(tokens)
    event_mask = [0]*len(tokens)
    # Create dict for [out_prefix]_entities.json
    for mention in entity_mentions:
        for pos in range(mention['start'], mention['end']+1):
          entity_mask[pos] = 1
  
        entities.append({'doc_id': doc_id,
                         'subtopic': '0',
                         'm_id': '0',
                         'sentence_id': sent_id,
                         'tokens_ids': list(range(mention['start'], mention['end']+1)),
                         'tokens': ' '.join(tokens[mention['start']:mention['end']+1]),
                         'tags': '',
                         'lemmas': '',
                         'cluster_id': '0',
                         'cluster_desc': '',
                         'singleton': False})

    # Create dict for [out_prefix]_events.json
    for mention in event_mentions: 
        try: # XXX
          start = mention['trigger']['start']
          end = mention['trigger']['end']
        except:
          start = mention['start']
          end = mention['end']

        for pos in range(start, end+1):
          event_mask[pos] = 1

        events.append({'doc_id': doc_id,
                       'subtopic': '0',
                       'm_id': '0',
                       'sentence_id': sent_id,
                       'tokens_ids': list(range(start, end)),
                       'tokens': ' '.join(tokens[start:end]),
                       'tags': '',
                       'lemmas': '',
                       'cluster_id': '0',
                       'cluster_desc': '',
                       'singleton': False})

    # Create dict for [out_prefix].json
    if not doc_id in outs: 
      outs[doc_id] = []
    
    for idx, token in enumerate(tokens):
      outs[doc_id].append([sent_id, idx, token, entity_mask[idx] > 0 or event_mask[idx] > 0])
  json.dump(outs, codecs.open(out_prefix+'.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(entities, codecs.open(out_prefix+'_entities.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(events, codecs.open(out_prefix+'_events.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(entities+events, codecs.open(out_prefix+'_mixed.json', 'w', 'utf-8'), indent=4, sort_keys=True)

if __name__ == '__main__':
  data_dir = 'data/video_m2e2'
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    os.mkdir(os.path.join(data_dir, 'mentions'))
    os.mkdir(os.path.join(data_dir, 'gold'))
  data_json = 'm2e2/data/video_m2e2/grounding_video_m2e2_small.json'
  out_prefix = os.path.join(data_dir, 'mentions/train')
  get_mention_doc(data_json, out_prefix) 
