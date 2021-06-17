import json
import codecs
import collections
import pandas as pd
import seaborn as sns; sns.set()
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve 
import numpy as np
import argparse
import pyhocon
from conll import write_output_file

def create_type_stoi(mention_jsons):
  n_entity_types = 0
  n_event_types = 0
  type_to_idx = {'###NULL###':0}
  for mention_json in mention_jsons:
    mention_dicts = json.load(open(mention_json))
    for mention_dict in mention_dicts:
      if 'entity_type' in mention_dict:
        entity_type = mention_dict['entity_type']
        if not entity_type in type_to_idx:
          type_to_idx[entity_type] = len(type_to_idx)
          n_entity_types += 1
      else:
        event_type = mention_dict['event_type']
        if not event_type in type_to_idx:
          type_to_idx[event_type] = len(type_to_idx)
          n_event_types += 1
  print('Num. of entity types = {}'.format(n_entity_types))
  print('Num. of event types = {}'.format(n_event_types))
  return type_to_idx

def create_role_stoi(mention_jsons):
  roles = set()
  for mention_json in mention_jsons:
    mention_dicts = json.load(open(mention_json))
    for mention_dict in mention_dicts:
      if 'event_type' in mention_dict:
        for a in mention_dict['arguments']:
          role = f"{mention_dict['event_type']}.{a['role']}"
          if not role in roles:
            roles.add(role)
  roles = sorted(roles, key=lambda x:x.split('.')[0])
  roles = ['###NULL###'] + roles
  role_to_idx = {r:i for i, r in enumerate(roles)}
  print('Num. of role types = {}'.format(len(role_to_idx)))
  return role_to_idx

def create_ontology(event_jsons, entity_jsons, out_prefix):
  event_stoi = create_type_stoi(event_jsons)
  entity_stoi = create_type_stoi(entity_jsons)
  role_stoi = create_role_stoi(event_jsons)
  ontology = {'entities': sorted(entity_stoi, key=lambda x:entity_stoi[x]),
              'event': sorted(event_stoi, key=lambda x:event_stoi[x]),
              'arguments': sorted(role_stoi, key=lambda x:role_stoi[x])}
  json.dump(ontology, open(f'{out_prefix}.json', 'w'), indent=2)
  
def create_feature_stoi(mention_jsons, feature_types):
  stoi = dict()
  for mention_json in mention_jsons:
    mentions = json.load(open(mention_json))
    for feat_type in feature_types:
      if not feat_type in stoi:
        stoi[feat_type] = dict()
      for m in mentions:
        if not m[feat_type] in stoi[feat_type]:
          if isinstance(m[feat_type], int) or isinstance(m[feat_type], float):
            stoi[feat_type][m[feat_type]] = m[feat_type]
          else:
            stoi[feat_type][m[feat_type]] = len(stoi[feat_type])
  return stoi

def create_ie_to_srl_mapping(ontology_file, out_prefix):
  ie_roles = json.load(open(ontology_file))['arguments']
  cur_event = ''
  arg_idx = 0
  ie_to_srl = dict()
  for ie_role in ie_roles:
    event = ie_role.split('.')[0]
    role = re.sub(r"[0-9]", "", ie_role.split('.')[-1])
    if event != cur_event:
      cur_event = event
      arg_idx = 0
    if not role in ie_to_srl:
      ie_to_srl[role] = f"arg_{arg_idx}"
    arg_idx += 1
  json.dump(ie_to_srl, open(f'{out_prefix}.json', 'w'), indent=2)

def make_prediction_readable(pred_json, img_dir, mention_json, out_file='prediction_readable.txt'):
  pred_dicts = json.load(open(pred_json))
  mention_dicts = json.load(open(mention_json)) 
  f = codecs.open(out_file, 'w')
  for pred_dict in pred_dicts:
    doc_id = pred_dict['doc_id']
    tokens = pred_dict['tokens']
    spans = pred_dict['mention_spans']
    mention_texts = [' '.join(tokens[span[0]:span[1]+1]) for span in spans] # [label_dict[doc_id][span] for span in spans]    
    first = [mention_texts[m_idx] for m_idx in pred_dict['first_idx']]
    second = [mention_texts[m_idx] for m_idx in pred_dict['second_idx']]
    pairwise_label = pred_dict['pairwise_label']
    score = pred_dict['score']
    for a, b, l, s in zip(first, second, pairwise_label, score):
      f.write('{}\t{}\t{}\t{:d}\t{:.2f}\n'.format(doc_id, a, b, l, s))
  f.close()

def make_prediction_readable_crossmedia(pred_json, out_file='prediction_readable.txt'):
  pred_dicts = json.load(open(pred_json))
  f = codecs.open(out_file, 'w')
  for doc_idx, pred_dict in enumerate(pred_dicts):
    doc_id = pred_dict['doc_id']
    tokens = pred_dict['tokens']
    spans = pred_dict['mention_spans']
    image_labels = pred_dict['image_labels']
    mention_texts = [' '.join(tokens[span[0]:span[1]+1]) for span in spans] # [label_dict[doc_id][span] for span in spans]    
    first = [mention_texts[m_idx] for m_idx in pred_dict['first_idx'][0]] # XXX
    second = [image_labels[m_idx] for m_idx in pred_dict['second_idx'][0]] # XXX
    pairwise_label = pred_dict['pairwise_label'][0] # XXX
    score = pred_dict['score']
    for a, b, l, s in zip(first, second, pairwise_label, score):
      f.write('{}\t{}\t{}\t{:d}\t{:.2f}\n'.format(doc_id, a, b, l, s))
  f.close()

def plot_pr_curve(pred_json, model_name='Multimedia Coref.'):
  pred_dicts = json.load(open(pred_json))
  y_score = []
  y_test = []
  for p in pred_dicts:
    y_score.extend(p['score'])
    if len(np.asarray(p['pairwise_label']).shape) == 2:
      p['pairwise_labels'] = p['pairwise_label'][0]
    y_test.extend(p['pairwise_labels'])
  

  average_precision = average_precision_score(y_test, y_score)
  print('Average_precision: {}'.format(average_precision))
  precision, recall, thresholds = precision_recall_curve(y_test, y_score)
  f1s = 2. * precision * recall / np.maximum(precision + recall, 1e-10)
  print('Best threshold: {}, best F1: {}'.format(thresholds[np.argmax(f1s)], f1s.max()))
  df = {'Model': [model_name]*len(precision),
        'Precision': precision.tolist(),
        'Recall': recall.tolist()}
  return df

def save_gold_conll_files(doc_json, mention_json, dir_path):
  if not os.path.isdir(dir_path):
    os.mkdir(dir_path)
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
    if len(start_ends) == 0:
      starts = start_ends
      ends = start_ends
      doc_ids = []
    else:
      starts = start_ends[:, 0]
      ends = start_ends[:, 1]
      doc_ids = [doc_id]*len(cur_label_dict)

    # Extract clusters
    clusters = collections.defaultdict(list)
    for m_idx, span in enumerate(sorted(cur_label_dict)):
      cluster_id = cur_label_dict[span]
      clusters[cluster_id].append(m_idx)
    non_singletons = {}
    non_singletons = {cluster: ms for cluster, ms in clusters.items() if len(ms) > 1}
    doc_name = doc_id
    write_output_file({doc_id:document}, non_singletons, doc_ids, starts, ends, dir_path, doc_name)

def plot_trigger_type_across_datasets(mention_jsons, dataset_names):
  trigger_to_type = dict()
  for dset_name, mention_json in zip(dataset_names, mention_jsons):
    mentions = json.load(open(mention_json))
    for m in mentions:
      trigger = m['head_lemma']
      event_type = m['event_type']

      if not trigger in trigger_to_type:
        trigger_to_type[trigger] = {dset_name:[event_type]}
      elif not dset_name in trigger_to_type[trigger]:
        trigger_to_type[trigger][dset_name] = [event_type]
      else:
        trigger_to_type[trigger][dset_name].append(event_type)
      
  json.dump(trigger_to_type, 
            open('trigger_info_across_datasets.json', 'w'), 
            indent=2, 
            sort_keys=True)         

def compare_ontology_across_datasets(type_by_dataset_file, dataset1, dataset2):
  type_by_dataset = json.load(open(type_by_dataset_file, 'r'))
  stoi = {dataset1: dict(), dataset2: dict()}
  n_triggers1 = 0
  n_triggers2 = 0
  n_overlap_triggers = 0
  for token in type_by_dataset:
    labels1 = type_by_dataset[token].get(dataset1, [])
    labels2 = type_by_dataset[token].get(dataset2, [])
    for y1 in labels1:
      if not y1 in stoi[dataset1]:
        stoi[dataset1][y1] = len(stoi[dataset1]) 
    
    for y2 in labels2:
      if not y2 in stoi[dataset2]:
        stoi[dataset2][y2] = len(stoi[dataset2])

  n_types1 = len(stoi[dataset1])
  n_types2 = len(stoi[dataset2])
  confusion = np.zeros((n_types1, n_types2))
  for token in type_by_dataset:
    labels1 = type_by_dataset[token].get(dataset1, [])
    labels2 = type_by_dataset[token].get(dataset2, [])
    if len(labels1) * len(labels2) > 0:
      n_overlap_triggers += 1
    
    if len(labels1) > 0:
      n_triggers1 += 1
    if len(labels2) > 0:
      n_triggers2 += 1

    for label1 in labels1:
      for label2 in labels2:
        y1 = stoi[dataset1][label1]
        y2 = stoi[dataset2][label2]
        confusion[y1, y2] += 1 

  confusion_norm = confusion / np.maximum(confusion.sum(1, keepdims=True), 1.)
  new_row_order = sorted(list(range(n_types1)),
                         key=lambda x:confusion_norm[x].max(),
                         reverse=True)
  confusion = confusion[new_row_order]
  confusion_norm = confusion_norm[new_row_order]

  new_col_order = []
  for i in range(n_types1):
    if i >= n_types2:
      break
    max_s = 0
    max_j = -1
    for j in range(n_types2):
      if confusion[i, j] >= max_s and not j in new_col_order:
        max_s = confusion[i, j]
        max_j = j
    new_col_order.append(max_j)

  for j in range(n_types2):
    if not j in new_col_order:
      new_col_order.append(j)

  fig, ax = plt.subplots(figsize=(8, 8))
  plt.pcolor(confusion_norm[:, new_col_order], cmap=plt.cm.Blues)
  plt.colorbar()
  plt.xlabel(dataset2)
  plt.ylabel(dataset1)
  
  '''
  for x in range(len(stoi[dataset2])):
    for y in range(len(stoi[dataset1])):
      if confusion[y, new_col_order[x]] > 0:
        plt.text(x, y, confusion[y, new_col_order[x]], fontsize=5, color='orange')
  '''

  xticks = np.arange(len(stoi[dataset2]))+0.5
  yticks = np.arange(len(stoi[dataset1]))+0.5
  ax.set_xticks(xticks)
  ax.set_yticks(yticks)

  xticklabels = sorted(stoi[dataset2], key=lambda x:stoi[dataset2][x])
  xticklabels = [xticklabels[i] for i in new_col_order]
  yticklabels = sorted(stoi[dataset1], key=lambda x:stoi[dataset1][x])
  yticklabels = [yticklabels[i] for i in new_row_order]
  ax.set_xticklabels(xticklabels, fontsize=8, rotation='vertical')
  ax.set_yticklabels(yticklabels, fontsize=8, rotation=45)
  ax.invert_yaxis()

  plt.savefig('ontology_comparison')
  plt.show()
  plt.close()

  print(f'Number of triggers in {dataset1}: {n_triggers1}\n'
        f'Number of triggers in {dataset2}: {n_triggers2}\n'
        f'Number of overlapping triggers: {n_overlap_triggers}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='')
  parser.add_argument('--task', type=int)
  args = parser.parse_args()

  config = pyhocon.ConfigFactory.parse_file(args.config)
  model_dir = config['model_path']
  # img_dir = config['image_dir']
  data_dir = os.path.join(config['data_folder'], '../')
  exp_dir = model_dir

  if args.task == 0: 
    prefix = args.config.split('/')[-1].split('.')[0]
    pred_jsons = ['{}_prediction_text_only_coref.json'.format(prefix), '{}_prediction_text_coref.json'.format(prefix)]
    for pred_json in pred_jsons:
      pred_json = os.path.join(exp_dir, pred_json)
      mention_json = os.path.join(data_dir, 'mentions/test_mixed.json')
      make_prediction_readable(pred_json, img_dir, mention_json, pred_json.split('.')[0]+'_readable.txt')   
      # make_prediction_readable_crossmedia(pred_json, pred_json.split('.')[0]+'_readable.txt')   
  elif args.task == 1:
    data_dir = config['data_folder']
    out_prefix = os.path.join(data_dir, 'train')
    save_gold_conll_files(out_prefix+'.json', out_prefix+'_mixed.json', os.path.join(data_dir, '../gold_mixed')) 
    save_gold_conll_files(out_prefix+'.json', out_prefix+'_events.json', os.path.join(data_dir, '../gold_events')) 
    save_gold_conll_files(out_prefix+'.json', out_prefix+'_entities.json', os.path.join(data_dir, '../gold_entities'))
  elif args.task == 2:
    prefix = args.config.split('/')[-1].split('.')[0]
    pred_jsons = ['{}_prediction_text_only_coref.json'.format(prefix), '{}_prediction_text_coref.json'.format(prefix)]
    model_names = ['Text RoBERTa', 'Multimedia RoBERTa']
    df = {'Model':[], 'Precision':[], 'Recall':[]}
    for pred_json, model_name in zip(pred_jsons, model_names):
      cur_df = plot_pr_curve(os.path.join(exp_dir, pred_json), model_name)
      df['Model'].extend(cur_df['Model'])
      df['Precision'].extend(cur_df['Precision'])
      df['Recall'].extend(cur_df['Recall'])

    df = pd.DataFrame(df)
    sns.lineplot(data=df, x='Recall', y='Precision', hue='Model')
    plt.savefig(os.path.join(exp_dir, 'precision_recall.png'))
    plt.show()
  elif args.task == 3:
    mention_json = 'test_create_type_to_idx.json'
    json.dump([{'entity_type': '1'},
               {'event_type': '2'},
               {'entity_type': '1'},
               {'entity_type': '1'}], open(mention_json, 'w'))
    type_to_idx = create_type_to_idx([mention_json])
    print(type_to_idx.items())
  elif args.task == 4:
    mention_jsons = ['data/ace/mentions/train_events.json',
                     'data/ace/mentions/dev_events.json',
                     'data/ace/mentions/test_events.json',
                     'data/video_m2e2/mentions/train_events.json',
                     'data/video_m2e2/mentions/test_events.json',]
    dset_names = ['ACE', 'ACE', 'ACE', 'Video M2E2', 'Video M2E2']
    plot_trigger_type_across_datasets(mention_jsons, dset_names)
  elif args.task == 5:
    compare_ontology_across_datasets('trigger_info_across_datasets.json', 'ACE', 'Video M2E2') 
  elif args.task == 6:
    out_prefix = 'data/video_m2e2/ontology_oneie'
    create_ontology(['data/video_m2e2/mentions/train_events.json',
                     'data/video_m2e2/mentions/test_events.json'],
                    ['data/video_m2e2/mentions/train_entities.json',
                     'data/video_m2e2/mentions/test_entities.json'],
                    out_prefix)
    create_ie_to_srl_mapping('data/video_m2e2/ontology_oneie.json',
                             'data/video_m2e2/ie_to_srl')
