# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import os
import collections
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
np.random.seed(2)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=20)
plt.rc('figure', titlesize=30)
plt.rc('font', size=20)

def visualize_image_features(embed_file,
                             label_file,
                             ontology_file=None,
                             freq_file=None,
                             label_type='event',
                             out_prefix='image_tsne',
                             n_class=10):
  """ Visualize the embeddings with TSNE """
  if not os.path.exists(f'{out_prefix}.csv'):
    if label_file.split('.')[-1] == 'npz':
      assert ontology_file is not None and freq_file is not None
      ontology = json.load(open(ontology_file))
      freq = json.load(open(freq_file))
      label_types = ontology[label_type]
      label_npz = np.load(label_file)
      labels = [label_types[y] for k in sorted(label_npz, key=lambda x:int(x.split('_')[-1]))
                for y in np.argmax(label_npz[k], axis=1)]
      top_types = [label_types[int(k)] for k in sorted(freq, key=lambda x:freq[x], reverse=True)[:n_class]]
    else:
      label_dict = json.load(open(label_file))
      labels = [y for k in sorted(label_dict) for y in label_dict[k]]
      freq = dict()
      for y in labels:
        if not y in freq:
          freq[y] = 1
        else:
          freq[y] += 1
      top_types = sorted(freq, key=lambda x:freq[x], reverse=True)[:n_class]

    feat_npz = np.load(embed_file)
    feats = np.concatenate([feat_npz[k] for k in sorted(feat_npz, key=lambda x:int(x.split('_')[-1]))]) # XXX

    X = TSNE(n_components=2).fit_transform(feats)
    select_idxs = [i for i, y in enumerate(labels) if str(y) in top_types]

    X = X[select_idxs]
    y = [labels[i] for i in select_idxs]
    print(X.shape, len(y))
    df = pd.DataFrame({'t-SNE dim 0': X[:, 0], 
                       't-SNE dim 1': X[:, 1],
                       'Event type': y})
    df.to_csv(out_prefix+'.csv')
  else:
    df = pd.read_csv(f'{out_prefix}.csv')

  fig, ax = plt.subplots(figsize=(10, 10))
  sns.scatterplot(data=df, x='t-SNE dim 0', y='t-SNE dim 1', 
                  hue='Event type', style='Event type',
                  palette=sns.color_palette('husl', len(top_types)))
  plt.savefig(out_prefix+'.png')
  plt.close()

def visualize_text_features(embed_file,
                            label_file,
                            label_type='event',
                            out_prefix='glove_tsne',
                            n_class=10):
  feat_npz = np.load(embed_file)
  label_dict = json.load(open(label_file, 'r'))
  feats = np.concatenate([feat_npz[k] for k in sorted(feat_npz, key=lambda x:int(x.split('_')[-1]))])
  tokens = [y[0] for k in sorted(label_dict, key=lambda x:int(x.split('_')[-1])) for y in label_dict[k]]
  labels = [y[1] for k in sorted(label_dict, key=lambda x:int(x.split('_')[-1])) for y in label_dict[k]]

  freq = {v:0 for v in set(labels)} 
  for y in labels:
    freq[y] += 1
  top_types = [k for k in sorted(freq, key=lambda x:freq[x], reverse=True)[:n_class]]
  stoi = {v:i for i, v in enumerate(top_types)}
  
  X = TSNE(n_components=2).fit_transform(feats)
  select_idxs = [i for i, y in enumerate(labels) if str(y) in top_types]

  X = X[select_idxs]
  y = [labels[i] for i in select_idxs]
  tokens = [tokens[i] for i in select_idxs]
  df = pd.DataFrame({'t-SNE dim 0': X[:, 0], 
                     't-SNE dim 1': X[:, 1],
                     f'{label_type} type': y})

  fig, ax = plt.subplots(figsize=(10, 10))
  plt.axis([min(X[:, 0])-1, max(X[:, 0])+1, min(X[:, 1])-1, max(X[:, 1])+1])
  palette = sns.color_palette('husl', len(top_types))
  for i in range(200):
    plt.text(X[i, 0], X[i, 1], 
             tokens[i], 
             fontsize=10, 
             color=palette[stoi[y[i]]])
  plt.savefig(f'{out_prefix}_text.png')
  plt.close()

  fig, ax = plt.subplots(figsize=(10, 10))
  sns.scatterplot(data=df, x='t-SNE dim 0', y='t-SNE dim 1', 
                  hue=f'{label_type} type', style=f'{label_type} type',
                  palette=palette)
  plt.savefig(f'{out_prefix}.png')
  plt.close()

def plot_result_vs_nclusters(result_csv):
  df = pd.read_csv(result_csv)
  fig, ax = plt.subplots(figsize=(10, 7))
  sns.lineplot(x='Number of Visual Clusters', y='Score', hue='Evaluation Metric', style='Evaluation Metric', data=df)
  # ax.set(xscale='log')
  ax.set_xticks(list(df['Number of Visual Clusters']))
  ax.set_xticklabels([str(x) for x in df['Number of Visual Clusters']])
  ax.yaxis.grid(True)
  plt.savefig(result_csv.split('.')[0]+'.png')
  plt.close()

def plot_attention(config, select_ids, 
                   text_only=False,
                   use_full_frame=False):
  EPS = 1e-10
  NULL = '##NULL##'
  root = config['root']
  data_path = config['data_folder']
  model_path = config['model_path']
  predictions = json.load(open(os.path.join(root, model_path, 'predictions.json')))
  events = json.load(open(os.path.join(root, data_path, 'test_events.json')))
  actions = json.load(open(os.path.join(root, data_path, '../master_readable.json')))
  event_dict = {doc_id:dict() for doc_id in select_ids}
  action_dict = {doc_id:dict() for doc_id in select_ids}
  for e in events:
    if e['doc_id'] in select_ids:
      span = (min(e['tokens_ids']), max(e['tokens_ids']))
      event_dict[e['doc_id']][span] = e['tokens']

  for k, a_info in actions.items():
    for a in a_info:
      if a['youtube_id'] in select_ids:
        span = a['Temporal_Boundary']
        dur = a['Total_Duration']
        span[0] = int(span[0] / dur * 100)
        span[1] = int(span[1] / dur * 100)
        action_dict[a['youtube_id']][tuple(span)] = a['Event_Type'].split('.')[-1] 

  for doc_id in select_ids:
    pred = None
    for pred in predictions:
      if pred['doc_id'] == doc_id:
        break
    
    fig, ax = plt.subplots(figsize=(10, 14))
    if use_full_frame:
      full_scores = prediction['score']
      scores = []
      for full_score in full_scores:
        score = []
        for span in sorted(action_dict[doc_id]):
          avg_score = full_score[span[0]:span[1]+1].mean(0, keepdims=True)
          score.append(avg_score)
        score.extend(full_score[100:])
        scores.append(score)
    else:
      scores = pred['score']

    e_tokens = [event_dict[doc_id][span] for span in sorted(event_dict[doc_id])]
    v_labels = [action_dict[doc_id][span] for span in sorted(action_dict[doc_id])]

    num_events = len(e_tokens)
    num_actions = len(v_labels)

    score_mat = []
    for score in scores:
      # XXX if text_only:
      #   pad = [0]*num_actions
      #   score = pad + score
      # if len(score) < num_events + num_actions + 1:
      if len(score) < num_events + 1:
        # XXX gap = num_events + num_actions + 1 - len(score)
        gap = num_events + 1 - len(score)
        score.extend([0]*gap)
        score_mat.append(score)

    score_mat = np.asarray(score_mat).T
    score_mat /= np.maximum(score_mat.sum(0), EPS)
    si = np.arange(num_events+1)
    # XXX ti = np.arange(num_events+num_actions+2)
    ti = np.arange(num_events+2)
    S, T = np.meshgrid(si, ti)
    plt.pcolormesh(S, T, score_mat, cmap=plt.cm.Blues)
    for i in range(num_events):
      # XXX for j in range(num_events+num_actions+1):
      for j in range(num_events+1):
        if score_mat[j, i]:
          plt.text(i+0.5, j+0.5, round(score_mat[j, i], 2), ha='center', color='orange')
    ax.set_xticks(si[1:]-0.5)
    ax.set_yticks(ti[1:]-0.5)
    ax.set_xticklabels(e_tokens)
    # XXX ax.set_yticklabels(v_labels+[NULL]+e_tokens)
    ax.set_yticklabels([NULL]+e_tokens) 
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(os.path.join(root, model_path, doc_id+'.png'))
    plt.close()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--task', type=int)
  parser.add_argument('--config', type=str)
  parser.add_argument('--mention_type', choices={'events', 'entities'}, default='events')
  args = parser.parse_args()

  if args.task == 0:
    data_dir = 'data/video_m2e2/mentions/' 
    out_prefix = os.path.join(data_dir, 'train_event_glove_embeddings')
    embed_file = f'{out_prefix}.npz'
    label_file = f'{out_prefix}_labels.json'
    visualize_text_features(embed_file,
                            label_file,
                            out_prefix=f'{out_prefix}_tsne')
  if args.task == 1:
    data_dir = 'data/video_m2e2/mentions/'
    label_type = 'Event' if args.mention_type == 'events' else 'Entity' 
    out_prefix = os.path.join(data_dir, f'train_oneie_events')
    embed_file = f'{out_prefix}.npz'
    label_file = f'{out_prefix}_labels.json'
    visualize_text_features(embed_file,
                            label_file,
                            label_type=label_type,
                            out_prefix=f'{out_prefix}_tsne')
  if args.task == 2:
    data_dir = 'data/video_m2e2/mentions/'
    # label_file = os.path.join(data_dir, 'test_mmaction_event_feat_labels_average.npz')
    embed_file = 'models/coref_crossmedia_events_video_m2e2/action_output.npz'
    label_file = 'models/coref_crossmedia_events_video_m2e2/action_class_labels.json'
    ontology_file = os.path.join(data_dir, '../ontology.json')
    freq_file =  os.path.join(data_dir, 'train_mmaction_event_feat_event_frequency.json')

    visualize_image_features(embed_file,
                             label_file,
                             ontology_file,
                             freq_file)
  if args.task == 3:
    config = json.load(open(args.config))
    select_ids = ['21Dgp1Zn-7w', '3XqUPrMEQx0', '69MJOTL3Sh8',
                  '6SQnvbd2AQc', 'FpMHb_-nrCQ', 'LHIbc7koTUE']
    # ['92PLcoWtn0Q', '9tx72NIbwh8', 'AohILHV6i8Q', 'GLOGR0UsBtk', 
    # 'LHIbc7koTUE', 'PaVqCYxGzp0', 'SvrpxITQ3Pk', 'dY_hkbVQA20', 
    # 'eaW-mv9IKOs', 'f3plTR1Dcew', 'fDm7S-pjpOo', 'fsYMznJdCok']
    if "visual" in config['modes']:
      plot_attention(config, select_ids)
    else:
      plot_attention(config, select_ids, text_only=True)
  if args.task == 4:
    result_csv = 'unsupervised/models/result_vs_nclusters/results.csv'
    plot_result_vs_nclusters(result_csv)
