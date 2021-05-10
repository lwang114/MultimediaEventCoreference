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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.special import logsumexp
import nltk
from nltk.stem import WordNetLemmatizer
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from coref.model_utils import pad_and_read_bert
from coref.utils import create_corpus

logger = logging.getLogger(__name__)
NULL = '###NULL###'
def extract_oneie_embeddings(embedding_file,
                             oneie_dir,
                             mention_json,
                             out_prefix):
  ''' Extract OneIE embeddings
  :param embedding_file: str, filename of embeddings of the format
      IEDs\t[mention id]\t[embedding vals separated by commas]
  :param oneie_dir: str, directory containing OneIE results
  :param mention_json: str, filename containing event and coreference annotation 
  '''
  lemmatizer = WordNetLemmatizer()

  label_dict = dict()
  mentions = json.load(open(mention_json))
  for m in mentions:
    if not m['doc_id'] in label_dict:
      label_dict[m['doc_id']] = dict()
    span = (min(m['tokens_ids']), max(m['tokens_ids']))
    label_dict[m['doc_id']][span] = m['cluster_id']

  # Create a dict from description id to span to oneie token ids and groundtruth mention info 
  label_dict_oneie = dict()
  documents = dict()
  for fn in os.listdir(oneie_dir):
    doc_id = '.'.join(fn.split('.')[:-1])
    if not doc_id in label_dict:
      continue
    label_dict_oneie[doc_id] = dict()
    documents[doc_id] = []
    start_sent = 0
    for line in open(os.path.join(oneie_dir, fn), 'r'):
      sent_dict = json.loads(line)
      sent_id = int(sent_dict['sent_id'].split('-')[-1])
      graph = sent_dict['graph']
      token_ids = sent_dict['token_ids']
      triggers_info = graph['triggers']
      entities_info = graph['entities']
      tokens = [t.lower() for t in sent_dict['tokens']]
      documents[doc_id].extend([[sent_id, start_sent+i, t, True] for i, t in enumerate(tokens)])
      postags = [t[1] for t in nltk.pos_tag(tokens, tagset='universal')]
      pos_abbrevs = [postag[0].lower() if postag in ['NOUN', 'VERB', 'ADJ'] else 'n' for postag in postags]
      roles_info = graph['roles']

      if not doc_id in label_dict_oneie:
        label_dict_oneie[doc_id] = dict()
      
      for trigger_idx, trigger in enumerate(triggers_info):
        trigger_tokens = tokens[trigger[0]:trigger[1]] 
        trigger_head_lemma = [lemmatizer.lemmatize(tokens[i], pos=pos_abbrevs[i]) for i in range(trigger[0], trigger[1])]
        
        span = (start_sent+trigger[0], start_sent+trigger[1]-1)
        mention_info = {'doc_id': doc_id,
                        'sentence_id': sent_id,
                        'tokens_ids': list(range(start_sent+trigger[0], start_sent+trigger[1])),
                        'token_ids': token_ids[trigger[0]:trigger[1]],
                        'tokens': trigger_tokens,
                        'head_lemma': ' '.join(trigger_head_lemma), 
                        'event_type': trigger[2],
                        'arguments': []} # XXX label_dict[doc_id][token_idx]

        for role in roles_info:
          if role[0] == trigger_idx:
            entity = entities_info[role[1]]
            head_lemma = lemmatizer.lemmatize(tokens[entity[1]-1].lower(), pos=pos_abbrevs[entity[1]-1])
            mention_info['arguments'].append({'start': start_sent+entity[0],
                                              'end': start_sent+entity[1],
                                              'tokens': ' '.join(tokens[entity[0]:entity[1]]),
                                              'token_ids': token_ids[entity[0]:entity[1]],
                                              'role': role[2],
                                              'head_lemma': head_lemma})
        label_dict_oneie[doc_id][span] = deepcopy(mention_info)
      start_sent += len(tokens)
  mentions = [label_dict_oneie[doc_id][span] for doc_id in sorted(label_dict_oneie) for span in sorted(label_dict_oneie[doc_id])]
  json.dump(documents, open(f'{out_prefix}.json', 'w'), indent=2)
  json.dump(mentions, open(f'{out_prefix}_events.json', 'w'), indent=2)

  # Create a mapping from token id to ([desc id]_[desc idx], span_idx) 
  token_id_to_emb = dict()
  for doc_idx, doc_id in enumerate(sorted(label_dict_oneie)):
    feat_id = f'{doc_id}_{doc_idx}'
    for span_idx, span in enumerate(sorted(label_dict_oneie[doc_id])):
      mention_info = label_dict_oneie[doc_id][span]
      for token_id in mention_info['token_ids']:
        token_id_to_emb[token_id] = (feat_id, span_idx, mention_info['tokens'], mention_info['event_type']) 

  embs = dict()
  event_labels = dict()
  with open(embedding_file, 'r') as f:
    for line in f:
      token, token_id, vec_str = line.strip().split('\t')
      doc_id = token_id.split(':')[0]
      if not doc_id in label_dict:
        continue
      if not token_id in token_id_to_emb:
        print(f'Token with id {token_id} is not a trigger')
        continue

      if doc_id == 'G0Cvgqj6CCI':
        print(token_id)
      feat_id, span_idx, token, event_type = token_id_to_emb[token_id]
      if not feat_id in embs:
        embs[feat_id] = []
        event_labels[feat_id] = []
      while span_idx >= len(embs[feat_id]):
        embs[feat_id].append([])
        event_labels[feat_id].append('')
      emb = [float(v) for v in vec_str.split(',')]
      embs[feat_id][span_idx].append(emb)
      event_labels[feat_id][span_idx] = [token, event_type]
    embs = {feat_id:np.stack([np.asarray(e).mean(axis=0) for e in embs[feat_id]]) for feat_id in embs}
  np.savez(f'{out_prefix}_events.npz', **embs)
  json.dump(event_labels, open(f'{out_prefix}_events_labels.json', 'w'), indent=2)

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

def extract_mention_glove_embeddings(config, split, glove_file, dimension=300, mention_type='events', out_prefix='event_glove_embedding', use_arguments=False):
    mention_json = os.path.join(config['data_folder'], f'{split}_{mention_type}.json')
    mentions = json.load(open(mention_json, 'r'))
    vocab = {'$$$UNK$$$': 0}
    label_dicts = dict()
    for m in mentions:
      span = (min(m['tokens_ids']), max(m['tokens_ids']))
      if not m['doc_id'] in label_dicts:
        label_dicts[m['doc_id']] = dict()
      label_dicts[m['doc_id']][span] = {'tokens': m['tokens'].split(),
                                        'head_lemma': m['head_lemma'],
                                        'arguments': m['arguments'] if 'arguments' in m else [], 
                                        'type': m['event_type'] if mention_type == 'events' else m['entity_type']}
      
      for token in m['tokens'].split():
        if not token in vocab:
          vocab[token] = len(vocab)
    
      if not m['head_lemma'] in vocab:
        vocab[m['head_lemma']] = len(vocab)
     
      if mention_type == 'events': 
        for a in m['arguments']:
          if not a['head_lemma'] in vocab:
            vocab[a['head_lemma']] = len(vocab)
          if not a['text'] in vocab:
            vocab[a['text']] = len(vocab)
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

    mention_embs = dict()
    labels = dict()
    for idx, doc_id in enumerate(sorted(label_dicts)):
      embed_id = f'{doc_id}_{idx}' 
      mention_embs[embed_id] = []
      labels[embed_id] = []  

      for span in sorted(label_dicts[doc_id]):
        head_lemma = label_dicts[doc_id][span]['head_lemma']  
        tokens = label_dicts[doc_id][span]['tokens']
        if head_lemma in vocab_emb:
          mention_emb = embed_matrix[vocab_emb.get(token, 0)]
        else: 
          mention_emb = np.asarray([embed_matrix[vocab_emb.get(token, 0)]\
                                    for token in tokens]).mean(0)   
        mention_type = label_dicts[doc_id][span]['type']

        if use_arguments:
          arg_emb = []
          n_args = len(label_dicts[doc_id][span]['arguments'])
          for a in label_dicts[doc_id][span]['arguments']:
            if a.get('head_lemma', '') in vocab_emb:
              a_token = a['head_lemma']
              arg_emb.append(np.asarray(embed_matrix[vocab_emb[a_token]]))
            else:
              a_tokens = a.get('text', '$$$UNK$$$').split()
              emb = []
              for a_token in a_tokens:
                emb.append(embed_matrix[vocab_emb.get(a_token, 0)])
              emb = np.asarray(emb).mean(0)
              arg_emb.append(emb)

          if len(arg_emb) > 10:
            arg_emb = arg_emb[:10]
          elif len(arg_emb) < 10:
            arg_emb.append(np.zeros(300*(10-len(arg_emb)))) 
          mention_emb = np.concatenate([mention_emb]+arg_emb)
        mention_embs[embed_id].append(mention_emb)
        labels[embed_id].append((token, mention_type)) 
      mention_embs[embed_id] = np.stack(mention_embs[embed_id])
    np.savez(out_prefix+'.npz', **mention_embs)
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

def extract_mention_cluster_probabilities(embed_files, 
                                          n_clusters=33,
                                          var=1):
    embed_npzs = [np.load(embed_file) for embed_file in embed_files]
    # if os.path.exists('mention_cluster_centroids.npy'):
    #   centroids = np.load('mention_cluster_centroids.npy')
    # else:
    kmeans = KMeans(n_clusters=n_clusters)
    X = np.concatenate([embed_npz[feat_id] for embed_npz in embed_npzs\
                       for feat_id in sorted(embed_npz, key=lambda x:int(x.split('_')[-1]))]) # XXX 
    centroids = kmeans.fit(X).cluster_centers_
    np.save('mention_cluster_centroids.npy', centroids)

    for embed_file, embed_npz in zip(embed_files, embed_npzs):
      cluster_probs = dict()
      for feat_id in sorted(embed_npz, key=lambda x:int(x.split('_')[-1])):
        cluster_probs[feat_id] = gaussian_softmax(embed_npz[feat_id], centroids, var)
      np.savez(embed_file.split('.')[0]+'_cluster_probs.npz', **cluster_probs)

def extract_mention_token_encodings(config,
                                    out_dir,
                                    mention_type='events',
                                    feat_type = 'head_lemma'):
  def to_one_hot(sent):
    K = len(vocab)
    sent = np.asarray(sent)
    if len(sent.shape) < 2:
      es = np.eye(len(vocab))
      sent = np.asarray([es[int(w)] if w < K else 1./K*np.ones(K) for w in sent])
      return sent
    else:
      return sent

  label_dict = dict()
  vocab = dict()
  for split in config['splits']:
    for dataset in config['splits'][split]:
      label_dict[dataset] = dict()
      mention_json = os.path.join(config['data_folder'], f'{dataset}_{mention_type}.json')
      mentions = json.load(open(mention_json, 'r'))
      for m in mentions:
        token = m[feat_type]
        span = (min(m['tokens_ids']), max(m['tokens_ids']))
        if not m['doc_id'] in label_dict[dataset]:
          label_dict[dataset][m['doc_id']] = dict()
        label_dict[dataset][m['doc_id']][span] = token
        if not token in vocab:
          vocab[token] = len(vocab)
  print(f'Vocabulary size: {len(vocab)}')

  for split in config['splits']:
    for dataset in config['splits'][split]:
      out_file = f'{out_dir}/{dataset}_{mention_type}_{feat_type}_labels.npz' 
      features = {}
      for feat_idx, doc_id in enumerate(sorted(label_dict[dataset])):
        features[f'{doc_id}_{feat_idx}'] = to_one_hot([vocab[label_dict[dataset][doc_id][span]] for span in sorted(label_dict[dataset][doc_id])])
      np.savez(out_file, **features)

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

def extract_entity_linguistic_features(config, split, out_prefix):
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
  entity_mentions = json.load(codecs.open(os.path.join(config['data_folder'], split+'_entities.json'), 'r', 'utf-8'))
  new_entity_mentions = []

  label_dict = dict()
  for m in entity_mentions:
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

  json.dump(new_entity_mentions, open(out_prefix+'_entities_with_linguistic_features.json', 'w'), indent=2)


def gaussian_softmax(X, centroids, var):
  T = X.shape[0]
  precision = 1. / var 
  # (T, K)
  logits = np.sum(-precision * (X[:, np.newaxis] - centroids[np.newaxis])**2, axis=-1) 
  # (T, K)
  log_probs = logits - logsumexp(logits, axis=1)[:, np.newaxis]
  return np.exp(log_probs)

def reduce_dim(embed_files, reduced_dim=300):
  embed_npzs = [np.load(embed_file) for embed_file in embed_files]
  pca = PCA(n_components=reduced_dim)
  X = np.concatenate([embed_npz[feat_id] for embed_npz in embed_npzs for feat_id in sorted(embed_npz, key=lambda x:int(x.split('_')[-1]))]) 
  pca.fit(X)
  print(f'Explained variance ratio: {sum(pca.explained_variance_ratio_)}')
  
  for embed_file, embed_npz in zip(embed_files, embed_npzs):
    embed_reduced = dict()
    for feat_id in sorted(embed_npz, key=lambda x:int(x.split('_')[-1])):
      embed_reduced[feat_id] = pca.transform(embed_npz[feat_id])
    embed_prefix = embed_file.split('.')[0]
    np.savez(f'{embed_prefix}_pca{reduced_dim}dim.npz', **embed_reduced)

def concat_embeddings(embed_files, out_prefix):
  embed_npzs = [np.load(embed_file) for embed_file in embed_files]
  feat_ids = sorted(np.load(embed_files[0]), key=lambda x:int(x.split('_')[-1]))
  new_embed_dict = dict()
  for feat_id in feat_ids:
    new_embed_dict[feat_id] = np.concatenate([embed_npz[feat_id] for embed_npz in embed_npzs], axis=1)
  np.savez(out_prefix+'.npz', **new_embed_dict)

def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str)
  parser.add_argument('--split', choices={'train', 'train_asr_sentence', 'test', 'train_unlabeled'})
  parser.add_argument('--mention_type', choices={'events', 'entities'}, default='events')
  parser.add_argument('--task', choices={'extract_glove_embeddings',
                                         'extract_bert_embeddings',
                                         'extract_entity_linguistic_features',
                                         'extract_event_linguistic_features',
                                         'extract_mention_glove_embeddings',
                                         'extract_mention_bert_embeddings',
                                         'extract_mention_cluster_probabilities',
                                         'reduce_dim',
                                         'concat_embeddings',
                                         'extract_mention_token_encodings',
                                         'extract_mention_glove_embeddings_with_arguments',
                                         'extract_visual_cluster_probabilities',
                                         'extract_oneie_embeddings'})
  args = parser.parse_args()  
  config = pyhocon.ConfigFactory.parse_file(args.config) 
  if not os.path.isdir(config['log_path']):
    os.makedirs(config['log_path']) 
  tasks = [args.task]
  glove_file = 'm2e2/data/glove/glove.840B.300d.txt'

  if args.task == 'extract_glove_embeddings':
    kwargs = {'config': config, 
              'split': args.split, 
              'glove_file': glove_file, 
              'out_prefix': '{}_glove_embeddings'.format(args.split)}
    extract_glove_embeddings(**kwargs)
  elif args.task == 'extract_bert_embeddings':
    kwargs = {'config': config, 
              'split': args.split, 
              'out_prefix': args.split}
    extract_bert_embeddings(**kwargs)
  elif args.task == 'extract_event_linguistic_features':
    for split in config['splits']:
      for dataset in config['splits'][split]:
        kwargs = {'config': config, 
                  'split': split, 
                  'out_prefix': split}
        extract_event_linguistic_features(**kwargs)
  elif args.task == 'extract_entity_linguistic_features':
    for split in config['splits']:
      for dataset in config['splits'][split]:
        kwargs = {'config': config, 
                  'split': split, 
                  'out_prefix': split}
        extract_entity_linguistic_features(**kwargs)
  elif args.task == 'extract_mention_glove_embeddings':
    for split in config['splits']:
      for dataset in config['splits'][split]:
        kwargs = {'config': config, 
                  'split': args.split, 
                  'glove_file': glove_file, 
                  'mention_type': args.mention_type, 
                  'out_prefix': f'{args.split}_{args.mention_type}_glove_embeddings'}
        extract_mention_glove_embeddings(**kwargs)
  elif args.task == 'extract_mention_bert_embeddings':
    kwargs = {'config': config,
              'split': args.split, 
              'mention_type': args.mention_type, 
              'out_prefix': f'{args.split}'}
    extract_mention_bert_embeddings(**kwargs)
  elif args.task == 'reduce_dim': 
    reduce_dim([os.path.join(config['data_folder'], f'{split}_events_roberta-large.npz')
                for split in ['train', 'test']], reduced_dim=300)
  elif args.task == 'concat_embeddings':
    kwargs = {'embed_files': [os.path.join(config['data_folder'], f'{args.split}_events_glove_embeddings.npz'), 
                              os.path.join(config['data_folder'], f'{args.split}_events_roberta-large_pca300dim.npz')],
              'out_prefix': os.path.join(config['data_folder'], f'{args.split}_events_glove_roberta-large_pca300dim')}
    concat_embeddings(**kwargs)
  elif args.task == 'extract_mention_cluster_probabilities':
    kwargs = {'embed_files': [os.path.join(config['data_folder'], f'{split}_events_with_arguments_glove_embeddings.npz')
                              for split in ['train', 'train_asr_sentence', 'test']],
              'n_clusters': 66}
    extract_mention_cluster_probabilities(**kwargs)
  elif args.task == 'extract_mention_token_encodings':
    extract_mention_token_encodings(config, config['data_folder'], feat_type='event_type')
  elif args.task == 'extract_mention_glove_embeddings_with_arguments':
    for split in config['splits']:
      for dataset in config['splits'][split]:
        kwargs = {'config': config, 
                  'split': dataset, 
                  'glove_file': glove_file, 
                  'mention_type': args.mention_type, 
                  'out_prefix': f'{dataset}_{args.mention_type}_with_arguments_glove_embeddings',
                  'use_arguments': True}
        extract_mention_glove_embeddings(**kwargs)
  elif args.task == 'extract_visual_cluster_probabilities':
    kwargs = {'embed_files': [os.path.join(config['data_folder'], f'{split}_mmaction_event_feat_average.npz')
                              for split in ['train', 'train_asr_sentence', 'test']],
              'n_clusters': 60}
    extract_mention_cluster_probabilities(**kwargs)
  elif args.task == 'extract_oneie_embeddings':
    embedding_file = os.path.join(config['data_folder'], '../m2e2_labeled_en.trigger.hidden.txt')
    oneie_dir = os.path.join(config['data_folder'], '../m2e2_labeled_oneie_result')
    for split in ['train', 'test']:
      mention_json = os.path.join(config['data_folder'], f'{split}_events.json')
      extract_oneie_embeddings(embedding_file,
                               oneie_dir,
                               mention_json,
                               out_prefix=os.path.join(config['data_folder'], f'{split}_oneie'))

if __name__ == '__main__':
  main()
