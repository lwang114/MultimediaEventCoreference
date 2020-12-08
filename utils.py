import json
import codecs
import collections
import pandas as pd
# import seaborn as sns; sns.set()
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve 


def make_prediction_readable(pred_json, img_dir, mention_json, out_file='prediction_readable.txt'):
  pred_dicts = json.load(open(pred_json))
  mention_dicts = json.load(open(mention_json))
  label_dict = collections.defaultdict(dict)
  for m in mention_dicts:
    if len(m['tokens_ids']) == 0:
      label_dict[m['doc_id']][(-1, -1)] = m['tokens']
    else:
      label_dict[m['doc_id']][(min(m['tokens_ids']), max(m['tokens_ids']))] = m['tokens']
  doc_ids = sorted(label_dict)
  # Filter doc_ids
  new_doc_ids = []
  for doc_id in doc_ids:
    if os.path.isfile(os.path.join(img_dir, doc_id+'.mp4')):
      new_doc_ids.append(doc_id)
  print('Keep {} out of {} documents'.format(len(new_doc_ids), len(doc_ids)))
  doc_ids = new_doc_ids

  f = codecs.open(out_file, 'w')
  for doc_id, pred_dict in zip(doc_ids, pred_dicts):
    spans = sorted(label_dict[doc_id])
    mention_texts = [label_dict[doc_id][span] for span in spans] # TODO Confirm
    first = [mention_texts[m_idx] for m_idx in pred_dict['first_idx']]
    second = [mention_texts[m_idx] for m_idx in pred_dict['second_idx']]
    pairwise_label = pred_dict['pairwise_label']
    score = pred_dict['score']
    for a, b, l, s in zip(first, second, pairwise_label, score):
      f.write('{}\t{}\t{}\t{:d}\t{:.2f}\n'.format(doc_id, a, b, l, s))
  f.close()

'''
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
  df = {'Model': [model_name]*len(precision),
        'Precision': precision.tolist(),
        'Recall': recall.tolist()}
  return df
'''

if __name__ == '__main__':
  model_dir = 'models/grounded_coref'
  img_dir = 'm2e2/data/video_m2e2/videos'
  data_dir = 'data/video_m2e2'
  pred_json = os.path.join(model_dir, 'prediction_multimedia_12_07_2020.json')
  mention_json = os.path.join(data_dir, 'mentions/test_mixed.json')
  make_prediction_readable(pred_json, img_dir, mention_json, pred_json.split('.')[0]+'_readable.txt')  
  '''
  exp_dir = '../pictures/12_1_2020'
  pred_jsons = ['prediction.json', 'prediction_multimedia.json']
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
  '''
  # plt.show()
