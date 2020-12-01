# -*- coding: utf-8 -*-

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
         'tokens_ids': list of ints, 1-indexed position of the tokens of the current mention in the sentences,
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
    doc_id = sen_dict['doc_id']
    sent_id = sen_dict['sent_id'].split('-')[-1]
    tokens = sen_dict['words']
    entity_mentions = sen_dict['golden-entity-mentions']    
    event_mentions = sen_dict['golden-event-mentions']

    entity_mask = np.zeros(len(tokens))
    event_mask = np.zeros(len(tokens))
    # Create dict for [out_prefix]_entities.json
    for mention in entity_mentions:
        entity_mask[mention['start']:mention['end']+1] = 1
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
        start = mention['trigger']['start']
        end = mention['trigger']['end']
        event_mask[start:end+1] = 1
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
      outs[doc_id].append([sent_id, idx+1, token, (entity_mask[idx] or event_mask[idx])]) # TODO Allow multiple sentences; currently assume one sentence per article
  json.dump(outs, open(out_prefix+'.json', 'w'), indent=4, sort_keys=True)
  json.dump(entities, open(out_prefix+'_entities.json', 'w'), indent=4, sort_keys=True)
  json.dump(events, open(out_prefix+'_events.json', 'w'), indent=4, sort_keys=True)
  json.dump(entities+events, open(out_prefix+'_mixed.json', 'w'), indent=4, sort_keys=True)

if __name__ == '__main__':
  data_json = '../m2e2/data/video_m2e2/grounding_video_m2e2_small.json'
  out_prefix = 'data/video_m2e2/train'
  get_mention_doc(data_json, out_prefix) 
