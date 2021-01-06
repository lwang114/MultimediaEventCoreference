from allennlp.predictors.predictor import Predictor
import allennlp_models.coref

class NeuralCoreferencer(nn.Module):
  def __init__(self):
    super(NeuralCoreferencer, self).__init__()
    # Initialize the coref model
    self.coref = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz')

  def forward(self):
    return # TODO

  def predict(self, tokens): # TODO Use gold span
    '''
    :param tokens:
    :return score:
    '''
    for child in self.coref.children():
      print(child)
      for p in child.parameters():
        print(p.size())
    instance = self.coref._dataset_reader.text_to_instance(tokens)
    labels = self.coref.predict_batch_instance(instance) 


if __name__ == '__main__':
  doc_json = 'data/video_m2e2/mentions/test.json'
  model = NeuralCoreferencer()
  documents = json.load(open(doc_json))
  for doc_id in sorted(documents)[9:10]:
    tokens = [token[2] for token in documents[doc_id]]
    print(model.predict(tokens))
 
