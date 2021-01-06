import torch
import torch.nn as nn
import json
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref

class NeuralCoreferencer(nn.Module):
  def __init__(self):
    super(NeuralCoreferencer, self).__init__()
    # Initialize the coref model
    self.predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz')
    self.coref = predictor._model
    
    for child_idx, child in enumerate(self.coref.children()): 
      for p in child.parameters():
        print(child_idx, p.size())

  def forward(self): # TODO Take existing embeddings as input
    return

  def predict(self, tokens, clusters): # TODO Use gold span
    '''
    :param tokens:
    :return score:
    '''
    instance = self.predictor._dataset_reader.text_to_instance(tokens)
    label_dict = self.predictor.predict_instance(instance)
    return label_dict

if __name__ == '__main__':
  data_dir = 'data/video_m2e2/mentions/'
  doc_json = os.path.join(data_dir, 'test.json')
  mention_json = os.path.join(data_dir, 'test_mixed.json')
  # For each doc, create a mapping from cluster id to span indices
  mentions = json.load(open(mention_json))
  cluster_dict = {}
  for m in mentions:
    start = min(m['token_ids'])
    end = max(m['token_ids'])
    if not m['doc_id'] in cluster_dict:
      cluster_dict[m['doc_id']] = {}

    if not m['cluster_id'] in cluster_dict[m['doc_id']]:
      cluster_dict[m['doc_id']][m['cluster_id']] = [[start, end]]
    else:
      cluster_dict[m['doc_id']][m['cluster_id']].append([start, end])

  model = NeuralCoreferencer()
  documents = json.load(open(doc_json))

  for doc_id in sorted(documents)[9:10]:
    clusters = [cluster_dict[doc_id][k] for k in sorted(cluster_dict[doc_id])]
    tokens = [token[2] for token in documents[doc_id]]
    # Predict coreference labels
    print(model.predict(tokens, clusters).items())
     
    # TODO Save outputs in CoNLL format

    # Save doc embeddings

