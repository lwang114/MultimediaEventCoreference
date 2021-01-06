import torch
import torch.nn as nn
import json
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref

class NeuralCoreferencer(nn.Module):
  def __init__(self):
    super(NeuralCoreferencer, self).__init__()
    # Initialize the coref model
    predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz')
    self.coref = predictor._model

  def forward(self):
    return # TODO

  def predict(self, tokens): # TODO Use gold span
    '''
    :param tokens:
    :return score:
    '''
    for child_idx, child in enumerate(self.coref.children()): 
      for p in child.parameters():
        print(child_idx, p.size())
    
    label_dict = self.coref.predict(tokens)
    return label_dict

if __name__ == '__main__':
  doc_json = 'data/video_m2e2/mentions/test.json'
  model = NeuralCoreferencer()
  documents = json.load(open(doc_json))
  for doc_id in sorted(documents)[9:10]:
    tokens = [token[2] for token in documents[doc_id]]
    print(model.predict(tokens).items())
 
