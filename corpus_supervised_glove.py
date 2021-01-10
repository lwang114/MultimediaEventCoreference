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

SINGLETON = '###SINGLETON###'
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
        if e >= max_token_num:
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
    pad = [torch.zeros(size, dtype=emb.dtype, device=emb.device).unsqueeze(0) for _ in range(L-emb.size(0))]
    emb = torch.cat([emb]+pad, dim=0)
  else:
    emb = emb[:L]
  return emb  

class SupervisedGroundingGloveFeatureDataset(Dataset):
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
    super(SupervisedGroundingGloveFeatureDataset, self).__init__()
    self.max_token_num = config.get('max_token_num', 512)
    self.max_span_num = config.get('max_span_num', 80)
    self.max_frame_num = config.get('max_frame_num', 20)
    self.max_mention_span = config.get('max_mention_span', 15)
    self.max_region_num = config.get('max_region_num', 20)
    self.img_feat_type = config.get('img_feat_type', 'resnet101')
    test_id_file = config.get('test_id_file', '')

    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    self.documents = documents
    self.origin_tokens = [documents[doc_id] for doc_id in sorted(documents)]
    text_mentions = json.load(codecs.open(text_mention_json, 'r', 'utf-8'))
    image_mentions = json.load(codecs.open(image_mention_json, 'r', 'utf-8'))
    
    # Extract image embeddings
    if 'image_feat_file' in config:
        self.imgs_embeddings = np.load(config.image_feat_file)
    else:
        print('{}_{}.npz'.format(doc_json.split('.')[0], self.img_feat_type))
        self.imgs_embeddings = np.load('{}_{}.npz'.format(doc_json.split('.')[0], self.img_feat_type))
       
    # Extract word embeddings
    glove_embed_file = '{}_glove_embeddings.npz'.format(doc_json.split('.')[0])
    self.docs_embeddings = np.load(glove_embed_file)
    
    # Extract coreference cluster labels
    self.text_label_dict, self.image_label_dict, image_token_dict = self.create_dict_labels(text_mentions, image_mentions)

    # Extract doc/image ids
    self.feat_keys = sorted(self.imgs_embeddings, key=lambda x:int(x.split('_')[-1]))[:20] # XXX
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

    # Extract spans
    self.candidate_start_ends = [np.asarray(sorted(self.text_label_dict[doc_id])) for doc_id in self.doc_ids]
    self.origin_candidate_start_ends = self.candidate_start_ends
    self.image_labels = [[image_token_dict[doc_id][box_id] for box_id in sorted(self.image_label_dict[doc_id], key=lambda x:x[0])] for doc_id in self.doc_ids]

  def create_dict_labels(self, text_mentions, image_mentions, clean_start_end_dict=None):
    '''
    :return text_label_dict: a mapping from doc id to a dict of (start token, end token) -> cluster id 
    :return image_label_dict: a mapping from image id to a dict of (bbox id, x min, y min, x max, y max) -> cluster id 
    '''
    cluster_dict = {SINGLETON: 0}
    text_label_dict = collections.defaultdict(dict)
    image_label_dict = {}
    image_token_dict = {} 
    for m in text_mentions:
      if len(m['tokens_ids']) == 0:
        text_label_dict[m['doc_id']][(-1, -1)] = 0
      else:
        start = min(m['tokens_ids'])
        end = max(m['tokens_ids'])
        if not m['cluster_id'] in cluster_dict:
            cluster_dict[m['cluster_id']] = len(cluster_dict)
        text_label_dict[m['doc_id']][(start, end)] = cluster_dict[m['cluster_id']]

    for i, m in enumerate(image_mentions):
        if not m['doc_id'] in image_label_dict:
            image_label_dict[m['doc_id']] = {}
            image_token_dict[m['doc_id']] = {}

        label_keys = [k[1] for k in sorted(image_label_dict[m['doc_id']], key=lambda x:x[0])]
        if not m['cluster_id'] in label_keys:
          bbox_id = len(image_label_dict[m['doc_id']])
          image_label_dict[m['doc_id']][(bbox_id, m['cluster_id'])] = cluster_dict.get(m['cluster_id'], 0)
          image_token_dict[m['doc_id']][(bbox_id, m['cluster_id'])] = m['tokens']
    return text_label_dict, image_label_dict, image_token_dict 
  
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
    origin_candidate_start_ends = self.origin_candidate_start_ends[idx]
    span_num = origin_candidate_start_ends.shape[0]
    candidate_start_ends = torch.LongTensor(origin_candidate_start_ends)
    candidate_start_ends = candidate_start_ends

    # Extract the current doc embedding
    doc_len = len(self.origin_tokens[idx])
    
    doc_embeddings = torch.FloatTensor(self.docs_embeddings[self.feat_keys[idx]])
    assert doc_len == doc_embeddings.shape[0]
    doc_embeddings = fix_embedding_length(doc_embeddings, self.max_token_num)

    start_mappings, end_mappings, continuous_mappings, width = get_all_token_mapping( 
                                       candidate_start_ends[:, 0],
                                       candidate_start_ends[:, 1],
                                       self.max_token_num,
                                       self.max_mention_span)

    width = torch.LongTensor([min(w, self.max_mention_span) for w in width])

    # Pad/truncate the outputs to max num. of spans
    start_mappings = fix_embedding_length(start_mappings, self.max_span_num)
    end_mappings = fix_embedding_length(end_mappings, self.max_span_num)
    continuous_mappings = fix_embedding_length(continuous_mappings, self.max_span_num)
    width = fix_embedding_length(width.unsqueeze(1), self.max_span_num).squeeze(1)
    
    # Extract coreference cluster labels
    labels = [int(self.text_label_dict[self.doc_ids[idx]][(start, end)]) for start, end in zip(origin_candidate_start_ends[:, 0], origin_candidate_start_ends[:, 1])]
    labels = torch.LongTensor(labels)
    labels = fix_embedding_length(labels.unsqueeze(1), self.max_span_num).squeeze(1)
    text_mask = torch.FloatTensor([1. if i < doc_len else 0 for i in range(self.max_token_num)])
    span_mask = torch.FloatTensor([1. if i < span_num else 0 for i in range(self.max_span_num)])
    # return start_end_embeddings, continuous_tokens_embeddings, width, labels, mask
    return doc_embeddings, start_mappings, end_mappings, continuous_mappings, width, labels, text_mask, span_mask

  def load_video(self, idx):
    '''Load video
    :param filename: str, video filename
    :return video_frames: FloatTensor of size (batch size, max num. of regions, image embed dim)
    :return mask: LongTensor of size (batch size, max num. of frames)
    '''    
    doc_id = self.doc_ids[idx]
    img_embeddings = self.imgs_embeddings[self.feat_keys[idx]]
    labels = [int(self.image_label_dict[doc_id][box_id]) for box_id in sorted(self.image_label_dict[doc_id], key=lambda x:int(x[0]))]
    region_num = len(labels)
    if not region_num == img_embeddings.shape[0] and region_num != 0:
      print('Number of labels not equal to the number of embeddings for {}: {} != {}'.format(doc_id, region_num, img_embeddings.shape[0])) 

    img_embeddings = torch.FloatTensor(img_embeddings)
    img_embeddings = img_embeddings.squeeze(-1).squeeze(-1)
    img_embeddings = fix_embedding_length(img_embeddings, self.max_region_num)

    labels = torch.LongTensor(labels)
    labels = fix_embedding_length(labels.unsqueeze(1), self.max_region_num).squeeze(1)
    mask = torch.FloatTensor([1. if i < region_num else 0 for i in range(self.max_region_num)])
    return img_embeddings, labels, mask

  def __getitem__(self, idx):
    img_embeddings, img_labels, video_mask = self.load_video(idx)
    doc_embeddings, start_mappings, end_mappings, continuous_mappings, width, text_labels, text_mask, span_mask = self.load_text(idx)
    return doc_embeddings, start_mappings, end_mappings, continuous_mappings, width, img_embeddings, text_labels, img_labels, text_mask, span_mask, video_mask

  def __len__(self):
    return len(self.doc_ids)
