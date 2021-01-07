import torch
import torch.nn as nn
import json
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import os 
from conll import write_output_file

class NeuralCoreferencer(nn.Module):
  def __init__(self):
    super(NeuralCoreferencer, self).__init__()
    # Initialize the coref model
    self.predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz')
    self.coref = self.predictor._model
    
    for child_idx, child in enumerate(self.coref.children()): 
      for p in child.parameters():
        print(child_idx, p.size())

  def forward(self): # TODO Take existing embeddings as input
    return

  def predict(self, sentences, gold_spans): # TODO Predict using user-specified span
    '''
    :param sentences: List of list of str,
    :param gold_spans:
    :return cluster_dict:
    '''
    output_dict = self.predictor.predict_tokenized(sentences)
    new_clusters = []
    for old_cluster in output_dict['clusters']:
      new_cluster = []
      for span in old_cluster:
        for gold_span in gold_spans:
          if gold_span[1] < span[0] or span[1] < gold_span[0]: 
            continue
          new_cluster.append(gold_span)
          break
      new_clusters.append(new_cluster)
    output_dict['clusters'] = new_clusters
    return output_dict

if __name__ == '__main__':
  data_dir = 'data/video_m2e2/mentions/'
  model_dir = 'models/coref_allennlp/'
  if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
    os.mkdir(os.path.join(model_dir, 'pred_conll'))

  doc_json = os.path.join(data_dir, 'test.json')
  mention_json = os.path.join(data_dir, 'test_mixed.json')
  # For each doc, create a mapping from cluster id to span indices
  mentions = json.load(open(mention_json))
  gold_span_dict = {}
  
  for m in mentions:
    start = min(m['tokens_ids'])
    end = max(m['tokens_ids'])
    if not m['doc_id'] in gold_span_dict:
      gold_span_dict[m['doc_id']] = [[start, end]]
    else:
      gold_span_dict[m['doc_id']].append([start, end])
  
  model = NeuralCoreferencer()
  documents = json.load(open(doc_json))

  for doc_id in sorted(documents):
    gold_spans = gold_span_dict[doc_id]
    sentences = [token[2] for token in documents[doc_id]]

    # Predict coreference labels
    output_dict = model.predict(sentences, gold_spans)
    pred_clusters = output_dict['clusters']

    # Save outputs in CoNLL format
    document = {doc_id:documents[doc_id]}
    starts = [span[0] for spans in pred_clusters for span in spans]
    ends = [span[1] for spans in pred_clusters for span in spans]
    span_dict = {(start, end):m_idx for m_idx, (start, end) in enumerate(zip(starts, ends))}
    predictions = {cluster_idx+1: [span_dict[tuple(span)] for span in spans] for cluster_idx, spans in enumerate(pred_clusters)}
    write_output_file(document, predictions, [doc_id]*len(starts),
                      starts, ends, os.path.join(model_dir, 'pred_conll'),
                      doc_id, False, True)

