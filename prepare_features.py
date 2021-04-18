import torch
import numpy as np
import argparse
import pyhocon
import os
import logging
import cv2
import codecs
import json
import math
from copy import deepcopy
import nltk
from nltk.stem import WordNetLemmatizer
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
# XXX
# import torchvision.transforms as transforms
import PIL.Image as Image
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from coref.model_utils import pad_and_read_bert
from coref.utils import create_corpus

logger = logging.getLogger(__name__)
NULL = '###NULL###'

def extract_glove_embeddings(config, split, glove_file, dimension=300, out_prefix='glove_embedding'):
    ''' Extract glove embeddings for a sentence
    :param doc_json: json metainfo file in m2e2 format
    :return out_prefix: output embedding for the sentences
    '''
    doc_json = os.path.join(config['data_folder'], split+'.json')
    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    '''
    if config.test_id_file and split == 'test':
        with open(config.test_id_file, 'r') as f:
            test_ids = sorted(f.read().strip().split('\n'), key=lambda x:int(x.split('_')[-1].split('.')[0]))
            test_ids = ['_'.join(k.split('_')[:-1]) for k in test_ids]
        documents = {test_id:documents[test_id] for test_id in test_ids}
    else:
        documents = cleanup(documents, config)
    '''
    print('Number of documents: {}'.format(len(documents)))
    vocab = {'$$$UNK$$$': 0}
    # Compute vocab of the documents
    for doc_id in sorted(documents): # XXX
        tokens = documents[doc_id]
        for token in tokens:
            if not token[2].lower() in vocab:
                # print(token[2].lower())
                vocab[token[2].lower()] = len(vocab)
    print('Vocabulary size: {}'.format(len(vocab)))
                
    embed_matrix = [[0.0] * dimension] 
    vocab_emb = {'$$$UNK$$$': 0} 
    # Load the embeddings
    with codecs.open(glove_file, 'r', 'utf-8') as f:
        for line in f:
            segments = line.strip().split()
            if len(segments) == 0:
                print('Empty line')
                break
            word = ' '.join(segments[:-300])
            if word in vocab:
                # print('Found {}'.format(word))
                embed= [float(x) for x in segments[-300:]]
                embed_matrix.append(embed)
                vocab_emb[word] = len(vocab_emb)
    print('Vocabulary size with embeddings: {}'.format(len(vocab_emb)))
    json.dump(vocab_emb, open(out_prefix+'_vocab.json', 'w'), indent=4, sort_keys=True)
    
    # Convert the documents into embedding sequence
    doc_embeddings = {}
    for idx, doc_id in enumerate(sorted(documents)): # XXX
        embed_id = '{}_{}'.format(doc_id, idx)
        # print(embed_id)
        tokens = documents[doc_id]
        doc_embedding = []
        for token in tokens:
            token_id = vocab_emb.get(token[2].lower(), 0)
            doc_embedding.append(embed_matrix[token_id])
        print(np.asarray(doc_embedding).shape)
        doc_embeddings[embed_id] = np.asarray(doc_embedding)
    np.savez(out_prefix+'.npz', **doc_embeddings)

def extract_bert_embeddings(config, split, out_prefix='bert_embedding'):
    device = torch.device('cuda:{}'.format(config.gpu_num[0]))
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    data = create_corpus(config, bert_tokenizer, split)
    doc_json = os.path.join(config['data_folder'], split+'.json')
    filtered_doc_ids = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    
    list_of_doc_id_tokens = []    
    for topic_num in tqdm(range(len(data.topic_list))):  
        list_of_doc_id_tokens_topic = [(doc_id, bert_tokens)
                                       for doc_id, bert_tokens in
                                       zip(data.topics_list_of_docs[topic_num], data.topics_bert_tokens[topic_num])]
        list_of_doc_id_tokens.extend(list_of_doc_id_tokens_topic)
    
    if 'test_id_file' in config and split == 'test':
        with open(config.test_id_file, 'r') as f:
            test_ids = f.read().strip().split('\n')
            test_ids = ['_'.join(k.split('_')[:-1]) for k in test_ids]
        list_of_doc_id_tokens = [doc_id_token for doc_id_token in list_of_doc_id_tokens if doc_id_token[0] in test_ids]
            
    list_of_doc_id_tokens = sorted(list_of_doc_id_tokens, key=lambda x:x[0]) # XXX
    print('Number of documents: {}'.format(len(list_of_doc_id_tokens)))

    emb_ids = {}
    docs_embeddings = {}
    with torch.no_grad():
        total = len(list_of_doc_id_tokens)
        for i in range(total):
            doc_id = list_of_doc_id_tokens[i][0]
            if not doc_id in emb_ids and doc_id in filtered_doc_ids:
                emb_ids[doc_id] = '{}_{}'.format(doc_id, len(emb_ids))
        
        nbatches = total // config['batch_size'] + 1 if total % config['batch_size'] != 0 else total // config['batch_size']
        for b in range(nbatches):
            start_idx = b * config['batch_size']
            end_idx = min((b + 1) * config['batch_size'], total)
            batch_idxs = list(range(start_idx, end_idx))
            doc_ids = [list_of_doc_id_tokens[i][0] for i in batch_idxs]
            bert_tokens = [list_of_doc_id_tokens[i][1] for i in batch_idxs]
            bert_embeddings, docs_length = pad_and_read_bert(bert_tokens, bert_model)
            for idx, doc_id in enumerate(doc_ids):
                if not doc_id in emb_ids:
                    print('Skip {}'.format(doc_id))
                    continue
                emb_id = emb_ids[doc_id]
                bert_embedding = bert_embeddings[idx][:docs_length[idx]].cpu().detach().numpy()
                if emb_id in docs_embeddings:
                    print(doc_id, emb_id)
                    docs_embeddings[emb_id] = np.concatenate([docs_embeddings[emb_id], bert_embedding], axis=0)
                else:
                    docs_embeddings[emb_id] = bert_embedding
    np.savez(f"{out_prefix}_{config['bert_model']}.npz", **docs_embeddings)

def extract_type_embeddings(type_to_idx, glove_file):
    vocab_embs = sorted(type_to_idx, key=lambda x:type_to_idx[x])
    vocab_emb = {''}
    embed_matrix = [[0.0] * dimension] 
    # Load the embeddings
    with codecs.open(glove_file, 'r', 'utf-8') as f:
        for line in f:
            segments = line.strip().split()
            if len(segments) == 0:
                print('Empty line')
                break
            word = ' '.join(segments[:-300])
            if word in vocabs:
                # print('Found {}'.format(word))
                embed= [float(x) for x in segments[-300:]]
                embed_matrix.append(embed)
                vocab_emb[word] = len(vocab_emb)
    print('Vocabulary size with embeddings: {}'.format(len(vocab_emb)))



def extract_event_linguistic_features(config, split, out_prefix):
  def _head_word(phrase):
    postags = [t[1] for t in nltk.pos_tag(phrase, tagset='universal')]
    instance = dep_parser._dataset_reader.text_to_instance(phrase, postags)
    parsed_text = dep_parser.predict_batch_instance([instance])[0]
    head_idx = np.where(np.asarray(parsed_text['predicted_heads']) <= 0)[0][0]
    return phrase[head_idx], head_idx

  dep_parser = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
  # dep_parser._model = dep_parser._model.cuda()
  lemmatizer = WordNetLemmatizer() 

  doc_json = os.path.join(config['data_folder'], split+'.json')
  documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
  event_mentions = json.load(codecs.open(os.path.join(config['data_folder'], split+'_events.json'), 'r', 'utf-8'))
  new_event_mentions = []

  label_dict = dict()
  for m in event_mentions:
    span = (min(m['tokens_ids']), max(m['tokens_ids']))
    if not m['doc_id'] in label_dict:
      label_dict[m['doc_id']] = {}
    label_dict[m['doc_id']][span] = deepcopy(m) 

  for doc_id in sorted(documents): # XXX
    if not doc_id in label_dict:
      continue
    print(doc_id)
    tokens = [t[2] for t in documents[doc_id]]
    # Extract POS tags and word class
    wordclasses = [t[1] for t in nltk.pos_tag(tokens, tagset='universal')]
    postags = [t[1] for t in nltk.pos_tag(tokens)]

    for span_idx, span in enumerate(sorted(label_dict[doc_id])):
      new_mention = deepcopy(label_dict[doc_id][span])
      span_tokens = label_dict[doc_id][span]['tokens'].split()
      span_tags = postags[span[0]:span[1]+1]
    
      # Extract lemmatized head (HL)
      head, head_idx = _head_word(span_tokens) 
      head_class = wordclasses[span[0]+head_idx]
      pos_abbrev = head_class[0].lower() if head_class in ['NOUN', 'VERB', 'ADJ'] else 'n'
      new_mention['head_lemma'] = lemmatizer.lemmatize(head, pos=pos_abbrev)
      new_mention['pos_tag'] = postags[span[0]+head_idx]
      new_mention['word_class'] = head_class if head_class in ['NOUN', 'VERB', 'ADJ'] else 'OTHER' 

      # Extract the left and right lemmatized words of the head (LHL, RHL)
      if span[0] > 0:
        left_idx = span[0]-1 if wordclasses[span[0]-1] != '.' else span[0]-2
        left_class = wordclasses[left_idx] 
        pos_abbrev = left_class[0].lower() if left_class in ['NOUN', 'VERB', 'ADJ'] else 'n'
        left_word = lemmatizer.lemmatize(tokens[left_idx], pos=pos_abbrev)
      else:
        left_word = NULL
      
      if span[1] < len(tokens)-1:
        right_class = wordclasses[span[1]+1]
        pos_abbrev = right_class[0].lower() if right_class in ['NOUN', 'VERB', 'ADJ'] else 'n'
        right_word = lemmatizer.lemmatize(tokens[span[1]+1], pos=pos_abbrev)
      else:
        right_word = NULL

      new_mention['left_lemma'] = left_word
      new_mention['right_lemma'] = right_word

      # Extract the left and right lemmatized event mentions (LHE, RHE)
      if span_idx > 0:
        left_span = sorted(label_dict[doc_id])[span_idx-1]
        left_event = label_dict[doc_id][left_span]['tokens'].split()
        left_event_head, left_head_idx = _head_word(left_event)
        left_ev_class = wordclasses[left_span[0]+left_head_idx]
        pos_abbrev = left_ev_class[0].lower() if left_ev_class in ['NOUN', 'VERB', 'ADJ'] else 'n'
        new_mention['left_event_lemma'] = lemmatizer.lemmatize(left_event_head, pos=pos_abbrev)
      else:
        new_mention['left_event_lemma'] = NULL

      if span_idx < len(label_dict[doc_id]) - 1:
        right_span = sorted(label_dict[doc_id])[span_idx+1]
        right_event = label_dict[doc_id][right_span]['tokens'].split()
        right_event_head, right_head_idx = _head_word(right_event)
        right_ev_class = wordclasses[right_span[0]+right_head_idx]
        pos_abbrev = right_ev_class[0].lower() if right_ev_class in ['NOUN', 'VERB', 'ADJ'] else 'n'
        new_mention['right_event_lemma'] = lemmatizer.lemmatize(right_event_head, pos=pos_abbrev)
      else:
        new_mention['right_event_lemma'] = NULL
      new_event_mentions.append(new_mention)
  json.dump(new_event_mentions, open(out_prefix+'_events_with_linguistic_features.json', 'w'), indent=2)
    
def cleanup(documents, config):
    filtered_documents = {}
    img_ids = [img_id.split('.')[0] for img_id in os.listdir(config['image_dir'])]
    # config['image_dir'] = os.path.join(config['data_folder'], 'train_resnet152') # XXX
    img_files = os.listdir(config['image_dir'])

    if img_files[0].split('.')[-1] == '.jpg':
      img_ids = [img_id.split('.jpg')[0] for img_id in img_files]
    elif img_files[0].split('.')[-1] == '.npy':
      img_ids = ['_'.join(img_id.split('_')[:-1]) for img_id in os.listdir(config['image_dir'])]     

    for doc_id in sorted(documents): 
        filename = os.path.join(config['image_dir'], doc_id+'.mp4')
        if os.path.exists(filename):
            filtered_documents[doc_id] = documents[doc_id]
        elif doc_id in img_ids:
            filtered_documents[doc_id] = documents[doc_id]
    print('Keep {} out of {} documents'.format(len(filtered_documents), len(documents)))
    return filtered_documents


def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='configs/config_grounded.json')
  parser.add_argument('--split', choices={'train', 'test'}, default='train')
  parser.add_argument('--task', type=int)
  args = parser.parse_args()  
  config = pyhocon.ConfigFactory.parse_file(args.config) 
  if not os.path.isdir(config['log_path']):
    os.mkdir(config['log_path']) 
  tasks = [args.task]

  if 0 in tasks:
    glove_file = 'm2e2/data/glove/glove.840B.300d.txt'
    extract_glove_embeddings(config, args.split, glove_file, out_prefix='{}_glove_embeddings'.format(args.split))
  if 1 in tasks:
    extract_bert_embeddings(config, args.split, out_prefix=args.split)
  if 2 in tasks:
    extract_event_linguistic_features(config, args.split, out_prefix=args.split)

if __name__ == '__main__':
  main()
