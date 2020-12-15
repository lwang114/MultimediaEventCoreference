from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import json
import codecs

# Initialize the coref model
predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz')

# Load the documents
doc_json = 'data/video_m2e2/mentions/test.json'
documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
doc_ids = sorted(documents)
document = ' '.join([token[2] for token in documents[doc_ids[0]]])
print(document)

prediction = predictor.predict(
    document=document
)

# Extract the document embeddings


# Extract the mention embeddings and save the mention info

print(prediction)


