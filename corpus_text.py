import numpy as np
import collections
import codecs
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
import cv2
import os
import PIL.Image as Image
from transformers import AutoTokenizer, AutoModel

def get_all_token_mapping(start, end, max_token_num, max_mention_span):
    try:
      span_num = len(start)
    except:
      raise ValueError('Invalid type for start={}, end={}'.format(start, end))
    start_mappings = torch.zeros((span_num, max_token_num), dtype=torch.float) 
    end_mappings = torch.zeros((span_num, max_token_num), dtype=torch.float) 
    span_mappings = torch.zeros((span_num, max_mention_span, max_token_num), dtype=torch.float)  
    length = []
    for span_idx, (s, e) in enumerate(zip(start, end)):
        if e >= max_token_num or s >= max_token_num:
          continue
        start_mappings[span_idx, s] = 1.
        end_mappings[span_idx, e] = 1.
        for token_count, token_pos in enumerate(range(s, e+1)):
          if token_count >= max_mention_span:
            break
          span_mappings[span_idx, token_count, token_pos] = 1.
        length.append(e-s+1)
    return start_mappings, end_mappings, span_mappings, length

def fix_embedding_length(emb, L):
  size = emb.size()[1:]
  if emb.size(0) < L:
    pad = [torch.zeros(size, dtype=emb.dtype).unsqueeze(0) for _ in range(L-emb.size(0))]
    emb = torch.cat([emb]+pad, dim=0)
  else:
    emb = emb[:L]
  return emb  

class TextFeatureDataset(Dataset):
  def __init__(self, doc_json, mention_json, config, split='train'):
    '''
    :param doc_json: dict of 
        [doc_id]: list of [sent id, token id, token, is entity/event]
    :param mention_json: store list of dicts of:
        {'doc_id': str, document id,
         'subtopic': '0',
         'm_id': '0',
         'sentence_id': str, order of the sentence,
         'tokens_ids': list of ints, 1-indexed position of the tokens of the current mention in the sentences,
         'tokens': str, tokens concatenated with space,
         'tags': '',
         'lemmas': '',
         'cluster_id': '0',
         'cluster_desc': '',
         'singleton': boolean, whether the mention is a singleton}
    '''
    super(TextFeatureDataset, self).__init__()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.max_token_num = config.get('max_token_num', 512)
    self.max_span_num = config.get('max_span_num', 80)
    self.max_frame_num = config.get('max_frame_num', 500)
    self.max_mention_span = config.get('max_mention_span', 15)
    self.is_one_indexed = config.get('is_one_indexed', False)
    test_id_file = config.get('test_id_file', '')
    self.split = split
    self.config = config

    # Extract doc ids
    with open(doc_json, 'r') as f:
      documents = json.load(f)
  
    with open(mention_json, 'r') as f:
      mentions = json.load(f)
    
    # Extract BERT embeddings
    bert_embed_file = '{}_bert_embeddings.npz'.format(doc_json.split('.')[0])
    self.docs_embeddings = np.load(bert_embed_file)
    self.feat_keys = sorted(self.docs_embeddings, key=lambda x:int(x.split('_')[-1]))
    self.feat_keys = [k for k in self.feat_keys if '_'.join(k.split('_')[:-1]) in documents]
    self.doc_ids = ['_'.join(k.split('_')[:-1]) for k in self.feat_keys] 

    if test_id_file:
      with open(test_id_file) as f:
        test_ids = ['_'.join(k.split('_')[:-1]) for k in f.read().strip().split()]

      if split == 'train':
        self.doc_ids = [doc_id for doc_id in self.doc_ids if not doc_id in test_ids]
        self.feat_keys = [k for k in self.feat_keys if not '_'.join(k.split('_')[:-1]) in test_ids]
      else:
        self.doc_ids = [doc_id for doc_id in self.doc_ids if doc_id in test_ids]
        self.feat_keys = [k for k in self.feat_keys if '_'.join(k.split('_')[:-1]) in test_ids]
      assert len(self.doc_ids) == len(self.feat_keys)
    
    self.doc_ids = self.doc_ids # XXX
    self.feat_keys = self.feat_keys # XXX
    documents = {doc_id:documents[doc_id] for doc_id in self.doc_ids}
    self.documents = documents
    print('Number of documents: {}'.format(len(self.doc_ids)))
    
    # Tokenize documents and extract token spans after bert tokenization
    self.tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    self.origin_tokens, self.bert_tokens, self.bert_start_ends, clean_start_end_dict = self.tokenize(documents) 

    # Extract coreference cluster labels
    self.label_dict = self.create_dict_labels(mentions)

    # Extract original mention spans
    self.origin_candidate_start_ends = [np.asarray([[start, end] for start, end in sorted(self.label_dict[doc_id])]) for doc_id in self.doc_ids]
    for idx, doc_id in enumerate(self.doc_ids):
      for start_end in self.origin_candidate_start_ends[idx]:
        if np.max(start_end) >= len(clean_start_end_dict[doc_id]):
          print(doc_id, max(start_end), len(clean_start_end_dict[doc_id]))
    self.candidate_start_ends = [np.asarray([[clean_start_end_dict[doc_id][start-self.is_one_indexed],
                                              clean_start_end_dict[doc_id][end-self.is_one_indexed]] 
                                              for start, end in start_ends]) 
                                 for doc_id, start_ends in zip(self.doc_ids, self.origin_candidate_start_ends)]
  
  def tokenize(self, documents):
    '''
    Tokenize the sentences in BERT format. Adapted from https://github.com/ariecattan/coref
    '''
    docs_bert_tokens = []
    docs_start_end_bert = []
    docs_origin_tokens = []
    clean_start_end_dict = {}

    for doc_id in sorted(documents):
      tokens = documents[doc_id]
      bert_tokens_ids, bert_sentence_ids = [], []
      start_bert_idx, end_bert_idx = [], [] # Start and end token indices for each bert token
      original_tokens = [] 
      clean_start_end = -1 * np.ones(len(tokens), dtype=np.int)
      bert_cursor = -1
      for i, token in enumerate(tokens):
        sent_id, token_id, token_text, flag_sentence = token
        bert_token = self.tokenizer.encode(token_text, add_special_tokens=True)[1:-1]   
        if bert_token:
          bert_start_index = bert_cursor + 1
          bert_tokens_ids.extend(bert_token)
          start_bert_idx.append(bert_start_index)
          bert_cursor += len(bert_token)

          bert_end_index = bert_cursor
          end_bert_idx.append(bert_end_index)
          
          clean_start_end[i] = len(original_tokens)
          original_tokens.append([sent_id, token_id, token_text, flag_sentence])
      docs_bert_tokens.append(bert_tokens_ids)
      docs_origin_tokens.append(original_tokens)
      clean_start_end_dict[doc_id] = clean_start_end.tolist() 
      start_end = np.concatenate((np.expand_dims(start_bert_idx, 1), np.expand_dims(end_bert_idx, 1)), axis=1)
      docs_start_end_bert.append(start_end)

    return docs_origin_tokens, docs_bert_tokens, docs_start_end_bert, clean_start_end_dict

  def create_dict_labels(self, mentions):
    '''
    :return label_dict: a mapping from doc id to a dict of (start token, end token) -> cluster id 
    '''
    label_dict = collections.defaultdict(dict)
    for m in mentions:
      if len(m['tokens_ids']) == 0:
        label_dict[m['doc_id']][(-1, -1)] = m['cluster_id']
      else:
        start = min(m['tokens_ids'])
        end = max(m['tokens_ids']) 
        label_dict[m['doc_id']][(start, end)] = m['cluster_id']
    return label_dict    
  
  def __getitem__(self, idx):
    '''Load mention span embeddings for the document
    :param idx: int, doc index
    :return start_end_embeddings: FloatTensor of size (max num. spans, 2, span embed dim)
    :return continuous_tokens_embeddings: FloatTensor of size (max num. spans, max mention span, span embed dim)
    :return mask: FloatTensor of size (max num. spans,)
    :return width: LongTensor of size (max num. spans,)
    :return labels: LongTensor of size (max num. spans,) 
    '''
    # Extract the original spans of the current doc
    origin_candidate_start_ends = self.origin_candidate_start_ends[idx]
    candidate_starts = self.candidate_start_ends[idx][:, 0]
    candidate_ends = self.candidate_start_ends[idx][:, 1]
    span_num = candidate_starts.shape[0]
    # candidate_starts = torch.LongTensor(candidate_starts)
    # candidate_ends = torch.LongTensor(candidate_ends)    

    # Extract the current doc embedding
    doc_len = len(self.bert_tokens[idx])
    for k in self.docs_embeddings:
      if '_'.join(k.split('_')[:-1]) == '_'.join(self.feat_keys[idx].split('_')[:-1]):
        doc_embeddings = self.docs_embeddings[k][:doc_len]
        break
    doc_embeddings = torch.FloatTensor(doc_embeddings)
    doc_embeddings = fix_embedding_length(doc_embeddings, self.max_token_num)

    # Convert the original spans to the bert tokenized spans
    bert_start_ends = self.bert_start_ends[idx]
    bert_candidate_starts = bert_start_ends[candidate_starts, 0]
    bert_candidate_ends = bert_start_ends[candidate_ends, 1]
    
    start_mappings, end_mappings, continuous_mappings, width =\
     get_all_token_mapping(bert_candidate_starts,
                           bert_candidate_ends,
                           self.max_token_num,
                           self.max_mention_span)
    width = torch.LongTensor([min(w, self.max_mention_span) for w in width])

    # Pad/truncate the outputs to max num. of spans
    start_mappings = fix_embedding_length(start_mappings, self.max_span_num)
    end_mappings = fix_embedding_length(end_mappings, self.max_span_num)
    continuous_mappings = fix_embedding_length(continuous_mappings, self.max_span_num)
    width = fix_embedding_length(width.unsqueeze(1), self.max_span_num).squeeze(1)
 
    # Extract coreference cluster labels
    labels = [int(self.label_dict[self.doc_ids[idx]][(start, end)]) for start, end in zip(origin_candidate_start_ends[:, 0], origin_candidate_start_ends[:, 1])]
    labels = torch.LongTensor(labels)
    labels = fix_embedding_length(labels.unsqueeze(1), self.max_span_num).squeeze(1)
    text_mask = torch.FloatTensor([1. if j < doc_len else 0 for j in range(self.max_token_num)])
    span_mask = torch.FloatTensor([1. if i < span_num else 0 for i in range(self.max_span_num)])

    return doc_embeddings, start_mappings, end_mappings, continuous_mappings, width, labels, text_mask, span_mask

  def __len__(self):
    return len(self.doc_ids)
