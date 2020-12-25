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

SINGLETON = '###SINGLETON###'
def get_all_token_embedding(embedding, start, end):
    span_embeddings, length = [], []
    for s, e in zip(start, end):
        indices = torch.tensor(range(s, e + 1))
        span_embeddings.append(embedding[indices])
        length.append(len(indices))
    return span_embeddings, length

def fix_embedding_length(emb, L):
  size = emb.size()[1:]
  if emb.size(0) < L:
    pad = [torch.zeros(size, dtype=emb.dtype).unsqueeze(0) for _ in range(L-emb.size(0))]
    emb = torch.cat([emb]+pad, dim=0)
  else:
    emb = emb[:L]
  return emb  

class SupervisedGroundingFeatureDataset(Dataset):
  def __init__(self, doc_json, text_mention_json, image_mention_json, config, split='train'):
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
    super(SupervisedGroundingFeatureDataset, self).__init__()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.max_token_num = config.get('max_token_num', 512)
    self.max_span_num = config.get('max_span_num', 80)
    self.max_frame_num = config.get('max_frame_num', 20)
    self.max_mention_span = config.get('max_mention_span', 15)
    self.max_region_num = config.get('max_region_num', 20)
    self.img_feat_type = config.get('img_feat_type', 'resnet101')
    test_id_file = config.get('test_id_file', '')

    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    self.documents = documents
    text_mentions = json.load(codecs.open(text_mention_json, 'r', 'utf-8'))
    image_mentions = json.load(codecs.open(image_mention_json, 'r', 'utf-8'))
    
    # Extract image embeddings
    if 'image_feat_file' in config:
        self.imgs_embeddings = np.load(config.image_feat_file)
    else:
        self.imgs_embeddings = np.load('{}_{}.npz'.format(doc_json.split('.')[0], self.img_feat_type))
       
    # Extract word embeddings
    bert_embed_file = '{}_bert_embeddings.npz'.format(doc_json.split('.')[0])
    self.docs_embeddings = np.load(bert_embed_file)
    
    # Extract coreference cluster labels
    self.text_label_dict, self.image_label_dict = self.create_dict_labels(text_mentions, image_mentions)
    
    # Extract doc/image ids
    self.feat_keys = sorted(self.imgs_embeddings, key=lambda x:int(x.split('_')[-1])) # XXX
    self.feat_keys = [k for k in self.feat_keys if '_'.join(k.split('_')[:-1]) in self.text_label_dict and '_'.join(k.split('_')[:-1]) in self.image_label_dict]
    if test_id_file:
      with open(test_id_file) as f:
        test_ids = ['_'.join(k.split('_')[:-1]) for k in f.read().strip().split()]

      if split == 'train':
        self.feat_keys = [k for k in self.feat_keys if not '_'.join(k.split('_')[:-1]) in test_ids]
      else:
        self.feat_keys = [k for k in self.feat_keys if '_'.join(k.split('_')[:-1]) in test_ids]
    self.doc_ids = ['_'.join(k.split('_')[:-1]) for k in self.feat_keys]
    documents = {doc_id:documents[doc_id] for doc_id in self.doc_ids}
    print('Number of documents: ', len(self.doc_ids))

    # Tokenize documents and extract token spans after bert tokenization
    self.tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    self.origin_tokens, self.bert_tokens, self.bert_start_ends, clean_start_end_dict = self.tokenize(documents)

    # Extract spans
    self.origin_candidate_start_ends = [np.asarray(sorted(self.text_label_dict[doc_id])) for doc_id in self.doc_ids]
    self.candidate_start_ends = [np.asarray([[clean_start_end_dict[doc_id][start], clean_start_end_dict[doc_id][end]] for start, end in start_ends])
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
        bert_tokens_ids, end_bert_idx = [], []
        start_bert_idx, end_bert_idx = [], []
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

  def create_dict_labels(self, text_mentions, image_mentions, clean_start_end_dict=None):
    '''
    :return text_label_dict: a mapping from doc id to a dict of (start token, end token) -> cluster id 
    :return image_label_dict: a mapping from image id to a dict of (bbox id, x min, y min, x max, y max) -> cluster id 
    '''
    cluster_dict = {SINGLETON: 0}
    text_label_dict = {} # collections.defaultdict(dict)
    image_label_dict = {} # collections.defaultdict(dict)
    for m in text_mentions:
      if len(m['tokens_ids']) == 0:
        text_label_dict[m['doc_id']][(-1, -1)] = 0
      else:
        start = min(m['tokens_ids'])
        end = max(m['tokens_ids'])
        if not m['cluster_id'] in cluster_dict:
            cluster_dict[m['cluster_id']] = len(cluster_dict)
        if not m['doc_id'] in text_label_dict:
          text_label_dict[m['doc_id']] = {}
        text_label_dict[m['doc_id']][(start, end)] = cluster_dict[m['cluster_id']]

    for i, m in enumerate(image_mentions):
        if not m['doc_id'] in image_label_dict:
          image_label_dict[m['doc_id']] = {}

        if isinstance(m['bbox'], str):
          bbox_id = int(m['bbox'].split('_')[-1])
          image_label_dict[m['doc_id']][(bbox_id, m['bbox'])] = cluster_dict.get(m['cluster_id'], 0)
        else:
          label_keys = [k[1] for k in sorted(image_label_dict[m['doc_id']], key=lambda x:x[0])]
          if not m['cluster_id'] in label_keys:
            bbox_id = len(image_label_dict[m['doc_id']])
            image_label_dict[m['doc_id']][(bbox_id, m['cluster_id'])] = cluster_dict[m['cluster_id']]

    return text_label_dict, image_label_dict    
  
  def load_text(self, idx):
    '''Load mention span embeddings for the document
    :param idx: int, doc index
    :return start_end_embeddings: FloatTensor of size (max num. spans, 2, span embed dim)
    :return continuous_tokens_embeddings: FloatTensor of size (max num. spans, max mention span, span embed dim)
    :return mask: FloatTensor of size (max num. spans,)
    :return width: LongTensor of size (max num. spans,)
    :return labels: LongTensor of size (max num. spans,) 
    '''
    # Extract the original spans of the current doc
    origin_candidate_starts = self.origin_candidate_start_ends[idx][:, 0]
    origin_candidate_ends = self.origin_candidate_start_ends[idx][:, 1]
    candidate_starts = self.candidate_start_ends[idx][:, 0]
    candidate_ends = self.candidate_start_ends[idx][:, 1]

    # Convert the original spans to the bert tokenized spans
    bert_start_ends = self.bert_start_ends[idx]
    bert_candidate_starts = bert_start_ends[candidate_starts, 0]
    bert_candidate_ends = bert_start_ends[candidate_ends, 1]
    span_num = len(bert_candidate_starts)
    
    # Extract the current doc embedding
    doc_len = len(self.bert_tokens[idx])
    doc_embeddings = self.docs_embeddings[self.feat_keys[idx]][:doc_len]
    doc_embeddings = torch.FloatTensor(doc_embeddings)
    start_end_embeddings = torch.cat((doc_embeddings[bert_candidate_starts],
                                      doc_embeddings[bert_candidate_ends]), dim=1)
    continuous_tokens_embeddings, width = get_all_token_embedding(doc_embeddings, 
                                                                  bert_candidate_starts,
                                                                  bert_candidate_ends)
    continuous_tokens_embeddings = torch.stack([fix_embedding_length(emb, self.max_mention_span)
                                                for emb in continuous_tokens_embeddings], axis=0)
    width = torch.LongTensor([min(w, self.max_mention_span) for w in width])
       
    # Pad/truncate the outputs to max num. of spans
    start_end_embeddings = fix_embedding_length(start_end_embeddings, self.max_span_num)
    continuous_tokens_embeddings = fix_embedding_length(continuous_tokens_embeddings, self.max_span_num)
    width = fix_embedding_length(width.unsqueeze(1), self.max_span_num).squeeze(1)
    
    # Extract coreference cluster labels
    labels = [int(self.text_label_dict[self.doc_ids[idx]][(start, end)]) for start, end in zip(candidate_starts, candidate_ends)]
    labels = torch.LongTensor(labels)
    labels = fix_embedding_length(labels.unsqueeze(1), self.max_span_num).squeeze(1)
    mask = torch.FloatTensor([1. if i < span_num else 0 for i in range(self.max_span_num)])
    return start_end_embeddings, continuous_tokens_embeddings, width, labels, mask

  def load_video(self, idx):
    '''Load video
    :param idx: int
    :return img_embeddings: FloatTensor of size (batch size, max num. of regions, image embed dim)
    :return mask: LongTensor of size (batch size, max num. of frames)
    '''    
    doc_id = self.doc_ids[idx]
    img_embeddings = self.imgs_embeddings[self.feat_keys[idx]]
    img_embeddings = torch.FloatTensor(img_embeddings)
    img_embeddings = img_embeddings.permute(1, 0, 2, 3).flatten(start_dim=1).t()
    img_embeddings = fix_embedding_length(img_embeddings, self.max_region_num)

    labels = [int(self.image_label_dict[doc_id][box_id]) for box_id in sorted(self.image_label_dict[doc_id], key=lambda x:int(x[0]))]
    region_num = len(labels)
    labels = torch.LongTensor(labels)
    labels = fix_embedding_length(labels.unsqueeze(1), self.max_region_num).squeeze(1)
    mask = torch.FloatTensor([1. if i < region_num else 0 for i in range(self.max_region_num)])
    return img_embeddings, labels, mask

  def __getitem__(self, idx):
    img_embeddings, img_labels, video_mask = self.load_video(idx)
    start_end_embeddings, continuous_embeddings, width, text_labels, span_mask = self.load_text(idx)
    return start_end_embeddings, continuous_embeddings, width, img_embeddings, text_labels, img_labels, span_mask, video_mask

  def __len__(self):
    return len(self.doc_ids)
