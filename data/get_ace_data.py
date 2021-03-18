# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import os
import collections
import argparse

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

      entity_event_mask = np.zeros(len(tokens))
      for entity_mention in entity_mentions:
        mention_id = entity_mention['id']
        entity_id = '-'.join(mention_id.split('-')[:-1])
        if not entity_id in cluster_ids:
          cluster_ids[entity_id] = len(cluster_ids)
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
        entity_event_mask[start:end] = 1.
        entities.append(entity)

      for event_mention in event_mentions:
        mention_id = event_mention['id']
        event_id = '-'.join(mention_id.split('-')[:-1])
        if not event_id in cluster_ids:
          cluster_ids[event_id] = len(cluster_ids) 
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

      sent = [[sent_id, sen_start+t_idx, t, int(entity_event_mask[t_idx]) > 0] for t_idx, t in enumerate(tokens)]
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
      if i > 20:
        break
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
        entity_id = entity_mention['id']
        if not entity_id in cluster_ids:
          cluster_ids[entity_mention['id']] = len(cluster_ids)

        cluster_id = cluster_ids[entity_id]
        start = entity_mention['start']
        end = entity_mention['end']
        entity_type = entity_mention['entity_type']
        entities[entity_id] = {
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
        event_id = event_mention['id']
        if not event_id in cluster_ids:
          cluster_ids[event_mention['id']] = len(cluster_ids) 
        cluster_id = cluster_ids[event_mention['id']]
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
                 'm_id': event_id,
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


if __name__ == '__main__':
  data_dir = 'ace/'
  for split in ['train', 'dev', 'test']:
    data_json = os.path.join(data_dir, '{}.oneie.json'.format(split))
    out_prefix = os.path.join(data_dir, 'mentions', split)
    get_mention_doc(data_json, out_prefix)
    # get_event_info(data_json, out_prefix)
