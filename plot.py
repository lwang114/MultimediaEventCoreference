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
                             ontology_file,
                             freq_file,
                             label_type='event',
                             out_prefix='image_tsne',
                             n_class=10):
  """ Visualize the embeddings with TSNE """
  if not os.path.exists(f'{out_prefix}.csv'):
    ontology = json.load(open(ontology_file))
    label_types = ontology[label_type]
      
    feat_npz = np.load(embed_file)
    feats = np.concatenate([feat_npz[k] for k in sorted(feat_npz, key=lambda x:int(x.split('_')[-1]))]) # XXX
    label_npz = np.load(label_file)
    labels = [label_types[y] for k in sorted(label_npz, key=lambda x:int(x.split('_')[-1]))
              for y in np.argmax(label_npz[k], axis=1)]
    freq = json.load(open(freq_file))

    top_types = [label_types[int(k)] for k in sorted(freq, key=lambda x:freq[x], reverse=True)[:n_class]]
    X = TSNE(n_components=2).fit_transform(feats)
    select_idxs = [i for i, y in enumerate(labels) if str(y) in top_types]

    X = X[select_idxs]
    y = [labels[i] for i in select_idxs]
    df = pd.DataFrame({'t-SNE dim 0': X[:, 0], 
                       't-SNE dim 1': X[:, 1],
                       'Event type': y})
    df.to_csv(out_prefix+'.csv')
  else:
    df = pd.read_csv(f'{out_prefix}.csv')

  fig, ax = plt.subplots(figsize=(10, 10))
  sns.scatterplot(data=df, x='t-SNE dim 0', y='t-SNE dim 1', 
                  hue='Event type', style='Event type',
                  palette=sns.color_palette('husl', 10))
  plt.savefig(out_prefix+'.png')
  plt.close()

def visualize_glove_features(embed_file,
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
  df = pd.DataFrame({'t-SNE dim 0': X[:, 0], 
                     't-SNE dim 1': X[:, 1],
                     'Event type': y})

  fig, ax = plt.subplots(figsize=(10, 10))
  plt.axis([min(X[:, 0])-1, max(X[:, 0])+1, min(X[:, 1])-1, max(X[:, 1])+1])
  palette = sns.color_palette('husl', 10)
  for i in range(len(select_idxs)):
    plt.text(X[i, 0], X[i, 1], 
             tokens[i], 
             fontsize=3, 
             color=palette[stoi[y[i]]])
  plt.savefig(f'{out_prefix}_text.png')
  plt.close()

  fig, ax = plt.subplots(figsize=(10, 10))
  sns.scatterplot(data=df, x='t-SNE dim 0', y='t-SNE dim 1', 
                  hue='Event type', style='Event type',
                  palette=palette)
  plt.savefig(f'{out_prefix}.png')
  plt.close()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--task', type=int)
  args = parser.parse_args()

  if args.task == 0:
    data_dir = 'data/video_m2e2/mentions/' 
    out_prefix = os.path.join(data_dir, 'train_event_glove_embeddings')
    embed_file = f'{out_prefix}.npz'
    label_file = f'{out_prefix}_labels.json'
    visualize_glove_features(embed_file,
                             label_file,
                             out_prefix=f'{out_prefix}_tsne')
