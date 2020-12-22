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
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.max_token_num = config.get('max_token_num', 512)
    self.max_span_num = config.get('max_span_num', 80)
    self.max_frame_num = config.get('max_frame_num', 20)
    self.max_mention_span = config.get('max_mention_span', 15)
    self.max_region_num = config.get('max_region_num', 20)

    test_id_file = config.get('test_id_file', '')

    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))

    text_mentions = json.load(codecs.open(text_mention_json, 'r', 'utf-8'))
    image_mentions = json.load(codecs.open(image_mention_json, 'r', 'utf-8'))
    
    # Extract image embeddings
    if 'image_feat_file' in config:
        self.imgs_embeddings = np.load(config.image_feat_file)
    else:
        self.imgs_embeddings = np.load('{}_{}.npz'.format(doc_json.split('.')[0], self.img_feat_type))
        
    # Extract doc/image ids
    self.feat_keys = sorted(self.imgs_embeddings, key=lambda x:int(x.split('_')[-1]))
    self.doc_ids = ['_'.join(k.split('_')[:-1]) for k in self.feat_keys]

    if test_id_file:
      with open(test_id_file) as f:
        test_ids = ['_'.join(k.split('_')[:-1]) for k in f.read().strip().split()]

      if split == 'train':
        self.doc_ids = [doc_id for doc_id in self.doc_ids if not doc_id in test_ids]
        self.feat_keys = [k for k in self.feat_keys if not '_'.join(k.split('_')[:-2]) in test_ids]
      else:
        self.doc_ids = [doc_id for doc_id in self.doc_ids if doc_id in test_ids]
        self.feat_keys = [k for k in self.feat_keys if '_'.join(k.split('_')[:-2]) in test_ids]
      assert len(self.doc_ids) == len(self.feat_keys)
    
    self.doc_ids = self.doc_ids # XXX
    self.feat_keys = self.feat_keys # XXX
    print('Number of documents: ', len(self.doc_ids))

    # Extract word embeddings
    glove_embed_file = '{}_glove_embeddings.npz'.format(doc_json.split('.')[0])
    self.docs_embeddings = np.load(glove_embed_file)
    
    # Extract coreference cluster labels
    self.text_label_dict, self.image_label_dict = self.create_dict_labels(text_mentions, image_mentions)
    

  def create_dict_labels(self, text_mentions, image_mentions, clean_start_end_dict=None):
    '''
    :return text_label_dict: a mapping from doc id to a dict of (start token, end token) -> cluster id 
    :return image_label_dict: a mapping from image id to a dict of (bbox id, x min, y min, x max, y max) -> cluster id 
    '''
    cluster_dict = {SINGLETON: 0}
    text_label_dict = collections.defaultdict(dict)
    image_label_dict = collections.defaultdict(dict)
    for m in text_mentions:
      if len(m['tokens_ids']) == 0:
        text_label_dict[m['doc_id']][(-1, -1)] = 0
      else:
        start = min(m['tokens_ids'])
        end = max(m['tokens_ids'])
        if not m['cluster_id'] in cluster_dict:
            cluster_dict[m['cluster_id']] = len(cluster_dict)
        text_label_dict[m['doc_id']][(start, end)] = cluster_dict[m['cluster_id']]

    prev_id = ''
    bbox_id = 0
    for i, m in enumerate(image_mentions):
        if prev_id != m['doc_id']:
            prev_id = m['doc_id']
            bbox_id = 0
        image_label_dict[m['doc_id']][tuple([bbox_id]+m['bbox'])] = cluster_dict.get(m['cluster_id'], 0)
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
    candidate_starts = self.candidate_start_ends[idx][:, 0]
    candidate_ends = self.candidate_start_ends[idx][:, 1]
    span_num = len(candidate_starts)

    # Extract the current doc embedding
    doc_len = len(self.origin_tokens[idx])
    doc_embeddings = self.docs_embeddings[self.feat_keys[idx]]
    assert doc_len == doc_embeddings.shape[0]
    
    doc_embeddings = torch.FloatTensor(doc_embeddings)

    start_end_embeddings = torch.cat((doc_embeddings[candidate_starts],
                                      doc_embeddings[candidate_ends]), dim=1)
    continuous_tokens_embeddings, width = get_all_token_embedding(doc_embeddings, 
                                                                  candidate_starts,
                                                                  candidate_ends)
    # Pad/truncate the outputs to max num. of spans
    start_end_embeddings = fix_embedding_length(start_end_embeddings, self.max_span_num)
    continuous_tokens_embeddings = fix_embedding_length(continuous_tokens_embeddings, self.max_span_num) 
    # width = torch.LongTensor([min(w, self.max_mention_span) for w in width])
    # width = fix_embedding_length(width.unsqueeze(1), self.max_span_num).squeeze(1)
    
    # Extract coreference cluster labels
    labels = [int(self.text_label_dict[self.doc_ids[idx]][(start, end)]) for start, end in zip(origin_candidate_starts, origin_candidate_ends)]
    labels = torch.LongTensor(labels)
    labels = fix_embedding_length(labels.unsqueeze(1), self.max_span_num).squeeze(1)
    mask = torch.FloatTensor([1. if i < span_num else 0 for i in range(self.max_span_num)])
    return start_end_embeddings, continuous_tokens_embeddings, width, labels, mask
    # return doc_embeddings, labels, mask

  def load_video(self, idx):
    '''Load video
    :param filename: str, video filename
    :return video_frames: FloatTensor of size (batch size, max num. of regions, image embed dim)
    :return mask: LongTensor of size (batch size, max num. of frames)
    '''    
    img_embeddings = self.imgs_embeddings[self.feat_keys[idx]]
    img_embeddings = torch.FloatTensor(img_embeddings)
    img_embeddings = img_embeddings.permute(1, 0, 2, 3).flatten(start_dim=1).t()
    img_embeddings = fix_embedding_length(img_embeddings, self.max_region_num)

    labels = [int(self.image_label_dict[self.doc_ids[idx]][box_id]) for box_id in sorted(self.image_label_dict, key=lambda x:int(x[0]))]
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