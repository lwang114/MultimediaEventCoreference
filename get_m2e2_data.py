# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import os
import collections
import torchvision.transforms as transforms
import PIL.Image as Image

def get_mention_doc(data_json, bbox_json, out_prefix, inclusive=False):
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
  :param bbox_json: str of filename storing a mapping from image id to a dict of 
      {'role': {[entity type]: ["0" or "1", x min, y min, x max, y max]},
       'event_type': str of event type name} 
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
      4. [prefix]_bboxes.json: store list of dicts of:
        {'doc_id': str, 
         'bbox': list of ints, [x min, y min, x max, y max] of the bounding box,
         'tokens': str, tokens concatenated with space,
         'cluster_id': cluster id [sentence id]_[tokens]
         } 
  '''
  sen_dicts = json.load(open(data_json))
  bbox_dict = json.load(open(bbox_json))
  outs = {}
  image_dict = collections.defaultdict(list)
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
    doc_id = '_'.join(sen_dict['sentence_id'].split('_')[:-1])
    if len(sen_dict['image']) == 0:
      continue

    for image_filename in sen_dict['image']:
      image_id = image_filename.split('.')[0]
      if not image_id in image_dict:
        image_dict[doc_id].append(image_id)
    sent_id = sen_dict['sentence_id']
    tokens = sen_dict['words']
    entity_mentions = sen_dict['golden-entity-mentions']
    event_mentions = sen_dict.get('golden-event-mentions', [])
    
    if doc_id != cur_id:
      cur_id = doc_id
      sen_start = 0
    
    entity_mask = [0]*len(tokens)
    event_mask = [0]*len(tokens)
    # Create dict for [out_prefix]_entities.json
    for m_id, mention in enumerate(entity_mentions):
        for pos in range(mention['start'], mention['end']+end_inc):
          entity_mask[pos] = 1
        entity_type = mention['entity-type']
        cluster_id = '{}_{}'.format(sent_id, entity_type)
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
        event_type = mention['event-type']
        try: # XXX
          start = mention['trigger']['start']
          end = mention['trigger']['end']
        except:
          start = mention['start']
          end = mention['end']

        for pos in range(start, end+end_inc if not inclusive else end):
          event_mask[pos] = 1

        cluster_id = '{}_{}'.format(doc_id, event_type)
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
    
  # Create dict for [out_prefix]_bboxes.json
  bboxes = []
  for doc_id in sorted(image_dict):
    for image_id in image_dict[doc_id]:
      if image_id in bbox_dict:
        bbox_info = bbox_dict[image_id]
        event_type = bbox_info['event_type']
        cluster_id = '{}_{}'.format(doc_id, event_type)
        bboxes.append({'doc_id': doc_id,
                       'bbox': image_id,
                       'tokens': [],
                       'cluster_id': cluster_id})
          
  json.dump(outs, codecs.open(out_prefix+'.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(entities, codecs.open(out_prefix+'_entities.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(events, codecs.open(out_prefix+'_events.json', 'w', 'utf-8'), indent=4, sort_keys=True)
  json.dump(entities+events, codecs.open(out_prefix+'_mixed.json', 'w', 'utf-8'), indent=4, sort_keys=True)

def extract_image_embeddings(config, prefix):
    img_dir = config['image_dir']
    doc_json = os.path.join(config['data_folder'], prefix+'.json')
    bbox_json = os.path.join(config['data_folder'], prefix+'_bboxes.json')
    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))   
    image_mentions = json.load(codecs.open(bbox_json, 'r', 'utf-8'))
    image_label_dict = collections.defaultdict(dict)
    prev_id = ''
    for m in image_mentions:
      if prev_id != m['doc_id']:
        prev_id = m['doc_id']
      image_label_dict[m['doc_id']][m['bbox']] = m['cluster_id']

    doc_ids = sorted(documents)[:20] # XXX
    transform = transforms.Compose([
              transforms.Resize(256),
              transforms.RandomHorizontalFlip(),
              transforms.RandomCrop(224),
              transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))])
    image_model = ResNet101(device=torch.device('cuda'))
    image_feats = {}
    for idx, doc_id in enumerate(doc_ids):
      img_ids = sorted(image_label_dict[doc_id], key=lambda x:int(x.split('_')[-1]))
      for img_id in img_ids:
        # Load image
        img = Image.open('{}/{}.jpg'.format(img_dir, img_id))
        img = transform(img)

        # Extract visual features
        img_feat = image_model(img.unsqueeze(0)).cpu().detach().numpy()
        print(img_id, img_feat.shape) 
        feat_id = '{}_{}'.format(doc_id, idx)
        if not feat_id in image_feats:
          image_feats[feat_id] = [img_feat]
        else:
          image_feats[feat_id].append(img_feat)
    image_feats = {k:np.concatenate(v) for k, v in img_feat.items()}
    np.savez(os.path.join(config['data_folder'], prefix+'resnet101.npz'), **image_feats)

if __name__ == '__main__':
  data_dir = 'data/m2e2'
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    os.mkdir(os.path.join(data_dir, 'mentions'))
    os.mkdir(os.path.join(data_dir, 'gold'))
  data_json = 'm2e2/data/m2e2_rawdata/article_event.json'
  bbox_json = 'm2e2/data/m2e2_rawdata/image_event.json'
  out_prefix = os.path.join(data_dir, 'mentions/test') # XXX
  get_mention_doc(data_json, out_prefix, inclusive=True)
