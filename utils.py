import json
import codecs
import pandas as pd
import seaborn as sns; sns.set()
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve 

'''
def make_prediction_readable(pred_json, mention_json, out_file='prediction_readable.txt'):
  pred_dicts = json.load(open(pred_json))
  mention_dicts = json.load(open(mention_json))
  label_dict = {}
  for m in mention_dicts:
    if len(m['tokens_ids']) == 0:
      label_dict[m['doc_id']][(-1, -1)] = m['text'] # TODO Confirm
    else:
      label_dict[m['doc_id']][(min(m['tokens_ids']), max(m['tokens_ids']))] = m['text']
  doc_ids = sorted(label_dict)

  f = codecs.open(out_file, 'w')
  for doc_id, pred_dict in zip(doc_ids, pred_dicts):
    spans = sorted(label_dict[doc_id])
    mention_texts = [label_dict[span]['text'] for span in spans] # TODO Confirm
    first = [mention_texts[m_idx] for m_idx in pred_dict['first']]
    second = [mention_texts[m_idx] for m_idx in pred_dict['second']]
    pairwise_label = pred_dict['pairwise_label']
    score = pred_dict['score']
    for a, b, l, s in zip(first, second, pairwise_label, score):
      f.write('doc_id {} {} {:d} {:.2f}\n'.format(doc_id, a, b, l, s))
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

if __name__ == '__main__':
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

  # plt.show()
