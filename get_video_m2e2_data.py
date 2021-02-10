# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import os
import collections
from conll import write_output_file

from nltk.translate import bleu_score
from nltk.metrics.scores import precision, recall, f_measure 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
          n_event_cluster += 1
          event_cluster2id[cluster_id] = n_entity_cluster + n_event_cluster 

        for mention in coreference['events'][cluster_id]:
          event2coref[mention[1]] = event_cluster2id[cluster_id]
          print('doc id: {}, mention id: {}, event cluster id: {}, event cluster idx: {}'.format(doc_id, mention[1], cluster_id, event2coref[mention[1]])) # XXX

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
          print('doc id: {}, mention id: {}, entity cluster id: {}, entity cluster idx: {}'.format(doc_id, mention[1], cluster_id, entity2coref[mention[1]])) # XXX
      
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

def extract_visual_event_embeddings(data_dir, csv_dir, mapping_file, annotation_file, out_prefix='videom2e2_visual_event', k=5):
  ''' Extract visual event embeddings
  :param config: str, config file for the dataset,
  :param split: str, 'train' or 'test',
  :param in_feat_file: str, frame-level video feature file,
  :param mapping_file: str, filename of a mapping from doc id to video id
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
  ann_dict = json.load(open(annotation_file))
  id2desc = {v['id'].split('v=')[-1]:k for k, v in mapping_dict.items()}
  doc_ids = sorted(id2desc)
  is_csv_used = {fn:0 for fn in os.listdir(csv_dir)}

  event_feats = {}
  event_labels = {}
  event_frequency = {}
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

    # Extract event features
    event_label = []
    event_feat = []
    
    for event_dict in ann_dict[desc]:
      event_type = event_dict['Event_Type']
      event_label.append(event_type)
      if not event_type in event_frequency:
        event_frequency[event_type] = 1
      else:
        event_frequency[event_type] += 1 
      start_time, end_time = event_dict['Temporal_Boundary']
      start, end = start_time * 24 // 15, end_time * 24 // 15

      # Extract the top k subspace vectors
      event_feat.append(PCA(n_components=k).fit(frame_feats[start:end+1]).components_.flatten())
    
    event_feat = np.stack(event_feat, axis=0)
    feat_id = '{}_{}'.format(doc_id, idx)
    event_labels[feat_id] = np.asarray(event_label)
    event_feats[feat_id] = event_feat
  np.savez('{}_embeddings.npz'.format(out_prefix), **event_feats)

  # Convert the labels to one-hot
  n_event_types = max(int(t) for t in event_frequency) + 1
  event_labels_onehot = {k:np.eye(n_event_types)[l] for k, l in event_labels.items()} 
  np.savez('{}_labels.npz'.format(out_prefix), **event_labels_onehot)
  json.dump(event_frequency, open('{}_event_frequency.json'.format(out_prefix), 'w'), indent=4, sort_keys=True)

def visualize_features(embed_file, label_file, freq_file):
  # Visualize the embeddings with TSNE
  X = np.load(embed_file)
  y = np.load(label_file)
  freq = json.load(open(freq_file))

  top_types = sorted(freq, key=lambda x:freq[x], reverse=True)
  X_new = TSNE(n_components=5).fit_transform(X) # TODO

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

def compute_bleu_similarity(doc_json, mapping_file, out_prefix):
  mapping_dict = json.load(open(mapping_file))
  documents = json.load(open(doc_json))
  # Extract mapping from youtube id to short description
  id2shortdesc = {v['id'].split('v=')[-1]:k for k, v in mapping_dict.items()}

  # Extract mapping from youtube id to long description
  id2longdesc = {k:[token[2] for token in documents[k]] for k in id2shortdesc if k in documents}

  # Extract a list of document ids
  doc_ids = sorted(documents)
  doc_num = len(doc_ids)
  bleu_scores = np.zeros((doc_num, doc_num))
  out_f = open('{}_bleu.txt'.format(out_prefix), 'w')

  # Iterate through every pair of documents  
  for i, first_id in enumerate(doc_ids):
    for j, second_id in enumerate(doc_ids):
      # Extract the short and long descriptions of the pair 
      first_short_desc = id2shortdesc[first_id].replace('- BBC News', '').split()
      second_short_desc = id2shortdesc[second_id].replace('- BBC News', '').split()
      first_long_desc = id2longdesc[first_id]
      second_long_desc = id2longdesc[second_id]

      # Compute BLEU score
      # bleu_scores[i, j] = round(bleu_score.sentence_bleu([first_long_desc], 
      #                                              second_long_desc, 
      #                                              weights=[0.5, 0.5]), 4) 
      if j > i:
        bleu_score_short = round(bleu_score.sentence_bleu([first_short_desc], 
                                                    second_short_desc,
                                                    weights=[0.5, 0.5]), 4)
        prec = round(precision(set(first_short_desc), set(second_short_desc)), 4)
        rec = round(recall(set(first_short_desc), set(second_short_desc)), 4)
        f1 = round(f_measure(set(first_short_desc), set(second_short_desc)), 4)
        if bleu_score_short > 0:
          print(first_id, second_id, bleu_score_short, f1)
          out_f.write('{}, {}: BLEU score={:.2f}, BLEU score (short)={:.2f}\n'.format(first_id, second_id, bleu_scores[i, j], bleu_score_short))
          out_f.write('{}, {}: Precision (short)={:.2f}, Recall (short)={:.2f}, F1 (short)={:.2f}\n'.format(first_id, second_id, prec, rec, f1))
          out_f.write('{}: {}\n'.format(first_id, first_short_desc))
          out_f.write('{}: {}\n\n'.format(second_id, second_short_desc))

  np.save('{}_bleu.npy'.format(out_prefix), bleu_scores)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--task', type=int)
  args = parser.parse_args()

  data_dir = 'data/video_m2e2/mentions/'
  mapping_file = 'm2e2/data/video_m2e2/video_m2e2.json'
  test_desc_file = 'm2e2/data/video_m2e2/unannotatedVideos_textEventCount.json'
  data_json = 'm2e2/data/video_m2e2/grounding_video_m2e2.json'
  csv_dir = os.path.join(data_dir, 'mmaction_feat')
  
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    os.mkdir(os.path.join(data_dir, 'mentions'))
    os.mkdir(os.path.join(data_dir, 'gold'))

  if args.task == 0:  
    out_prefix = os.path.join(data_dir, 'train')
    get_mention_doc(data_json, out_prefix, inclusive=False) 
    data_json = 'm2e2/data/video_m2e2/grounding_video_m2e2_test.json'
    out_prefix = os.path.join(data_dir, 'test')
    get_mention_doc(data_json, out_prefix, inclusive=True)
  elif args.task == 1:
    save_gold_conll_files(out_prefix+'.json', out_prefix+'_mixed.json', os.path.join(data_dir, 'gold')) 
  elif args.task == 2:
    out_prefix = '{}/{}'.format(data_dir, csv_dir.split('/')[-1])
    extract_image_embeddings(data_dir, csv_dir, mapping_file, out_prefix)
  elif args.task == 3:
    train_test_split('{}/{}.npz'.format(data_dir, csv_dir.split('/')[-1]), os.path.join(data_dir, 'test.json'), mapping_file, out_prefix)
  elif args.task == 4:
    print('Extract BLEU scores between training document pairs')
    doc_json = '{}/train.json'.format(data_dir)
    out_prefix = '{}/train'.format(data_dir)
    compute_bleu_similarity(doc_json, mapping_file, out_prefix)
    print('Extract BLEU scores for test document pairs')
    doc_json = '{}/test.json'.format(data_dir)
    out_prefix = '{}/test'.format(data_dir)
    compute_bleu_similarity(doc_json, mapping_file, out_prefix)
  elif args.task == 5:
    # Random split
    out_dir = 'data/video_m2e2_random_split/mentions/'
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    documents = json.load(open(os.path.join(data_dir, 'train.json')))
    entity_mentions = json.load(open(os.path.join(data_dir, 'train_entities.json')))
    event_mentions = json.load(open(os.path.join(data_dir, 'train_events.json')))
    mixed_mentions = json.load(open(os.path.join(data_dir, 'train_mixed.json')))
    
    n_docs = len(documents)
    doc_ids = sorted(documents)
    n_train = int(0.75 * n_docs)
    train_idxs = np.random.permuation(n_docs)[:n_train]
    train_ids = [doc_ids[i] for i in train_idxs] 
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


