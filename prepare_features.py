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

def extract_mention_glove_embeddings(config, split, glove_file, dimension=300, mention_type='events', out_prefix='event_glove_embedding'):
    mention_json = os.path.join(config['data_folder'], f'{split}_{mention_type}.json')
    mentions = json.load(open(mention_json, 'r'))
    vocab = {'$$$UNK$$$': 0}
    label_dicts = dict()
    for m in mentions:
      # XXX token = m['head_lemma']
      token = m['tokens']
      span = (min(m['tokens_ids']), max(m['tokens_ids']))
      if not m['doc_id'] in label_dicts:
        label_dicts[m['doc_id']] = dict()
      label_dicts[m['doc_id']][span] = {'token': token,
                                        'type': m['event_type'] if mention_type == 'events' else m['entity_type']}
      if not token in vocab:
        vocab[token] = len(vocab)
    print(f'Vocabulary size: {len(vocab)}')
    
    embed_matrix = [[0.0] * dimension]
    vocab_emb = {'$$$UNK$$$': 0}
    with codecs.open(glove_file, 'r', 'utf-8') as f:
      for line in f:
        segments = line.strip().split()
        if len(segments) == 0:
          print('Empty line')
          break
        word = ' '.join(segments[:-300])
        if word in vocab:
          embed_matrix.append([float(x) for x in segments[-300:]])
          vocab_emb[word] = len(vocab_emb)
    print(f'Vocabulary size with embeddings: {len(vocab_emb)}')
    json.dump(vocab_emb, open(out_prefix+'_vocab_glove_emb.json', 'w'), indent=4, sort_keys=True)

    event_embs = dict()
    labels = dict()
    for idx, doc_id in enumerate(sorted(label_dicts)):
      embed_id = f'{doc_id}_{idx}' 
      event_embs[embed_id] = []
      labels[embed_id] = [] 
      
      for span in sorted(label_dicts[doc_id]):
        token = label_dicts[doc_id][span]['token']   
        event_type = label_dicts[doc_id][span]['type']
        event_embs[embed_id].append(embed_matrix[vocab_emb.get(token, 0)])
        labels[embed_id].append((token, event_type)) 
      event_embs[embed_id] = np.asarray(event_embs[embed_id])
    np.savez(out_prefix+'.npz', **event_embs)
    json.dump(labels, open(out_prefix+'_labels.json', 'w'), indent=2)

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

def extract_mention_bert_embeddings(config, split, mention_type='events', out_prefix='mention_bert_embedding'):
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    doc_embs = np.load(os.path.join(config['data_folder'], f"{split}_{config['bert_model']}.npz"))
    doc_to_emb = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in sorted(doc_embs, key=lambda x:int(x.split('_')[-1]))}

    doc_json = os.path.join(config['data_folder'], split+'.json')
    mention_json = os.path.join(config['data_folder'], f'{split}_{mention_type}.json') 
    documents = json.load(open(doc_json))
    mentions = json.load(open(mention_json))
    label_dicts = dict()

    for m in mentions:
      token = m['tokens']
      span = (min(m['tokens_ids']), max(m['tokens_ids']))
      if not m['doc_id'] in label_dicts:
        label_dicts[m['doc_id']] = dict()
      label_dicts[m['doc_id']][span] = {'token': token,
                                        'type': m['event_type'] if mention_type == 'events' else m['entity_type']}

    event_embs = dict()
    labels = dict()
    for idx, doc_id in enumerate(label_dicts):
      tokens = documents[doc_id]
      clean_start_end = -1 * np.ones(len(tokens), dtype=np.int)
      bert_cursor = -1
      clean_tokens = []
      start_bert_idx, end_bert_idx = [], []
      for i, token in enumerate(tokens):
        sent_id, token_id, token_text, flag_sentence = token
        bert_token = bert_tokenizer.encode(token_text, add_special_tokens=True)[1:-1]
        if bert_token:
          bert_start_index = bert_cursor + 1
          start_bert_idx.append(bert_start_index)
          bert_cursor += len(bert_token)

          bert_end_index = bert_cursor
          end_bert_idx.append(bert_end_index)

          clean_start_end[i] = len(clean_tokens)
          clean_tokens.append(token)
      bert_spans = np.concatenate((np.expand_dims(start_bert_idx, 1), np.expand_dims(end_bert_idx, 1)), axis=1)

      doc_emb = doc_embs[doc_to_emb[doc_id]]  
      embed_id = f'{doc_id}_{idx}'
      event_embs[embed_id] = []
      labels[embed_id] = []
      for span in sorted(label_dicts[doc_id]):
        bert_span = (bert_spans[clean_start_end[span[0]]][0],\
                     bert_spans[clean_start_end[span[1]]][1])
        event_embs[embed_id].append(doc_emb[bert_span[0]:bert_span[1]+1].mean(axis=0))

        token = label_dicts[doc_id][span]['token']
        token_type = label_dicts[doc_id][span]['type']
        labels[embed_id].append((token, token_type))
      event_embs[embed_id] = np.stack(event_embs[embed_id])
    np.savez(f"{out_prefix}_{mention_type}_{config['bert_model']}.npz", **event_embs)
    json.dump(labels, open(f"{out_prefix}_{mention_type}_{config['bert_model']}_labels.json", 'w'), indent=2) 

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
      new_mention['head_lemma'] = lemmatizer.lemmatize(head.lower(), pos=pos_abbrev)
      new_mention['pos_tag'] = postags[span[0]+head_idx]
      new_mention['word_class'] = head_class if head_class in ['NOUN', 'VERB', 'ADJ'] else 'OTHER' 

      # Extract the left and right lemmatized words of the head (LHL, RHL)
      if span[0] > 0:
        left_idx = span[0]-1 if wordclasses[span[0]-1] != '.' else span[0]-2
        left_class = wordclasses[left_idx] 
        pos_abbrev = left_class[0].lower() if left_class in ['NOUN', 'VERB', 'ADJ'] else 'n'
        left_word = lemmatizer.lemmatize(tokens[left_idx].lower(), pos=pos_abbrev)
      else:
        left_word = NULL
      
      if span[1] < len(tokens)-1:
        right_class = wordclasses[span[1]+1]
        pos_abbrev = right_class[0].lower() if right_class in ['NOUN', 'VERB', 'ADJ'] else 'n'
        right_word = lemmatizer.lemmatize(tokens[span[1]+1].lower(), pos=pos_abbrev)
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
        new_mention['left_event_lemma'] = lemmatizer.lemmatize(left_event_head.lower(), pos=pos_abbrev)
      else:
        new_mention['left_event_lemma'] = NULL

      if span_idx < len(label_dict[doc_id]) - 1:
        right_span = sorted(label_dict[doc_id])[span_idx+1]
        right_event = label_dict[doc_id][right_span]['tokens'].split()
        right_event_head, right_head_idx = _head_word(right_event)
        right_ev_class = wordclasses[right_span[0]+right_head_idx]
        pos_abbrev = right_ev_class[0].lower() if right_ev_class in ['NOUN', 'VERB', 'ADJ'] else 'n'
        new_mention['right_event_lemma'] = lemmatizer.lemmatize(right_event_head.lower(), pos=pos_abbrev)
      else:
        new_mention['right_event_lemma'] = NULL
      
      # Extract argument head lemma
      for a_idx, a in enumerate(new_mention['arguments']):
        a_token = a['text'].split()
        a_head, a_head_idx = _head_word(a_token) 
        new_mention['arguments'][a_idx]['head_lemma'] = lemmatizer.lemmatize(a_head.lower())
      new_event_mentions.append(new_mention)
  json.dump(new_event_mentions, open(out_prefix+'_events_with_linguistic_features.json', 'w'), indent=2)
 
def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='configs/config_grounded.json')
  parser.add_argument('--split', choices={'train', 'test'}, default='train')
  parser.add_argument('--mention_type', choices={'events', 'entities'}, default='events')
  parser.add_argument('--task', type=int)
  args = parser.parse_args()  
  config = pyhocon.ConfigFactory.parse_file(args.config) 
  if not os.path.isdir(config['log_path']):
    os.mkdir(config['log_path']) 
  tasks = [args.task]
  glove_file = 'm2e2/data/glove/glove.840B.300d.txt'

  if 0 in tasks:
    extract_glove_embeddings(config, args.split, glove_file, out_prefix='{}_glove_embeddings'.format(args.split))
  if 1 in tasks:
    extract_bert_embeddings(config, args.split, out_prefix=args.split)
  if 2 in tasks:
    extract_event_linguistic_features(config, args.split, out_prefix=args.split)
  if 3 in tasks:
    extract_mention_glove_embeddings(config, args.split, glove_file, mention_type=args.mention_type, out_prefix=f'{args.split}_{args.mention_type}_glove_embeddings')
  if 4 in tasks:
    extract_mention_bert_embeddings(config, args.split, mention_type=args.mention_type, out_prefix=f'{args.split}')

if __name__ == '__main__':
  main()
