import json
import codecs
import collections
import pandas as pd
import seaborn as sns; sns.set()
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve 
import numpy as np
import argparse
import pyhocon

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

def plot_pr_curve(pred_json, model_name='Multimedia Coref.'):
  pred_dicts = json.load(open(pred_json))
  y_score = []
  y_test = []
  for p in pred_dicts:
    y_score.extend(p['score'])
    y_test.extend(p['pairwise_label'])

  average_precision = average_precision_score(y_test, y_score)
  print('Average_precision: {}'.format(average_precision))
  precision, recall, thresholds = precision_recall_curve(y_test, y_score)
  f1s = 2. * precision * recall / np.maximum(precision + recall, 1e-10)
  print('Best threshold: {}, best F1: {}'.format(thresholds[np.argmax(f1s)], f1s.max()))
  df = {'Model': [model_name]*len(precision),
        'Precision': precision.tolist(),
        'Recall': recall.tolist()}
  return df

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='')
  args = parser.parse_args()

  config = pyhocon.ConfigFactory.parse_file(args.config)
  model_dir = config['model_path']
  img_dir = config['image_dir']
  data_dir = os.path.join(config['data_folder'], '../')
  exp_dir = model_dir
  pred_jsons = ['{}_prediction.json'.format(args.config.split('/')[-1].split('.')[0])] # ['config_grounded_text_only_decode_prediction.json', 'config_grounded_prediction.json']
  for pred_json in pred_jsons:
    pred_json = os.path.join(exp_dir, pred_json)
    mention_json = os.path.join(data_dir, 'mentions/test_mixed.json')
    make_prediction_readable(pred_json, img_dir, mention_json, pred_json.split('.')[0]+'_readable.txt')   

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
  # plt.show()
