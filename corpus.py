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
PUNCT = [',', '.', '\'', '\"', ':', ';', '?', '!', '<', '>', '~', '%', '$', '|', '/', '@', '#', '^', '*']

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
    self.max_frame_num = config.get('max_frame_num', 100)
    self.max_mention_span = config.get('max_mention_span', 15)
    self.img_feat_type = config.get('video_feature', 'mmaction_feat')

    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    self.documents = documents
    text_mentions = json.load(codecs.open(text_mention_json, 'r', 'utf-8'))
    
    # Load action embeddings
    self.action_embeddings = np.load(os.path.join(config['data_folder'], f'{split}_mmaction_feat.npz'))
       
    # Load document embeddings
    bert_embed_file = '{}_{}.npz'.format(doc_json.split('.')[0], config.bert_model)
    self.docs_embeddings = np.load(bert_embed_file)
    
    # Extract coreference cluster labels
    self.text_label_dict, self.type_label_dict, self.mention_stoi = self.create_text_dict_labels(text_mentions)
    
    self.use_action_boundary = config['use_action_boundary']
    id_mapping_json = os.path.join(config['data_folder'], '../video_m2e2.json')
    action_anno_json = os.path.join(config['data_folder'], '../master.json') # Contain event time stamps
    action_dur_json = os.path.join(config['data_folder'], '../anet_anno.json')
    ontology_json = os.path.join(config['data_folder'], '../ontology.json')
    ontology_map_json = os.path.join(config['data_folder'], '../ontology_mapping.json')
    id_mapping = json.load(codecs.open(id_mapping_json, 'r', 'utf-8'))
    action_anno_dict = json.load(codecs.open(action_anno_json, 'r', 'utf-8'))
    action_dur_dict = json.load(codecs.open(action_dur_json))

    ontology_dict = json.load(codecs.open(ontology_json))
    if config['mention_type'] == 'event':
      ontology = ontology_dict['event']
    elif config['mention_type'] == 'entities':
      ontology = ontology_dict['entities']
    else:
      ontology = ontology_dict['event'] + ontology_dict['entities']
      self.ontology_map = json.load(codecs.open(ontology_map_json))
      self.action_label_dict, self.action_stoi = self.create_action_dict_labels(id_mapping,\
                                                                                action_anno_dict,\
                                                                                action_dur_dict,\
                                                                                ontology)

    # Extract doc/image ids
    self.feat_keys = sorted(self.action_embeddings, key=lambda x:int(x.split('_')[-1])) # XXX
    self.feat_keys = [k for k in self.feat_keys if '_'.join(k.split('_')[:-1]) in self.text_label_dict]
    self.doc_ids = ['_'.join(k.split('_')[:-1]) for k in self.feat_keys]
    documents = {doc_id:documents[doc_id] for doc_id in self.doc_ids}
    print('Number of documents: ', len(self.doc_ids))

    # Tokenize documents and extract token spans after bert tokenization
    self.tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    self.origin_tokens, self.bert_tokens, self.bert_start_ends, clean_start_end_dict = self.tokenize(documents)

    # Extract spans
    self.origin_candidate_start_ends = [np.asarray(sorted(self.text_label_dict[doc_id])) for doc_id in self.doc_ids]
    for doc_id, start_ends in zip(self.doc_ids, self.origin_candidate_start_ends):
      for start, end in start_ends:
        if end >= len(clean_start_end_dict[doc_id]):
          print(doc_id, start, end, len(clean_start_end_dict[doc_id])) 
    self.candidate_start_ends = [np.asarray([[clean_start_end_dict[doc_id][start], clean_start_end_dict[doc_id][end]] for start, end in start_ends])
                                 for doc_id, start_ends in zip(self.doc_ids, self.origin_candidate_start_ends)]
    self.image_labels = [[-1]*self.action_embeddings[feat_id].shape[0] for feat_id in self.feat_keys]

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
        bert_tokens_ids = []
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

  def create_text_dict_labels(self, text_mentions):
    '''
    :return text_label_dict: a mapping from doc id to a dict of (start token, end token) -> cluster id 
    :return image_label_dict: a mapping from image id to a dict of (bbox id, x min, y min, x max, y max) -> cluster id 
    '''
    text_label_dict = {}
    type_label_dict = {}
    stoi = {}
    for m in text_mentions:
      if len(m['tokens_ids']) == 0:
        text_label_dict[m['doc_id']][(-1, -1)] = 0
      else:
        start = min(m['tokens_ids'])
        end = max(m['tokens_ids'])
        if not m['doc_id'] in text_label_dict:
          text_label_dict[m['doc_id']] = {}
          type_label_dict[m['doc_id']] = {}

        if not m['event_type'] in stoi:
          stoi[m['event_type']] = len(stoi)
        text_label_dict[m['doc_id']][(start, end)] = m['cluster_id']
        type_label_dict[m['doc_id']][(start, end)] = m['event_type']

    return text_label_dict, type_label_dict, stoi
  
  def create_action_dict_labels(self, 
                               id_map,
                               anno_dict,
                               dur_dict, 
                               ontology):
    """ 
    :param id2desc: {[youtube id]: [description id with puncts]} 
    :param anno_dict: {[description id]: list of {'Temporal_Boundary': [float, float], 'Event_Type': int}}  
    :param dur_dict: {[description id]: {'duration_second': float}}
    :param ontology: list of mention classes
    :returns image_label_dict: {[doc_id]: {[action span]: [action class]}}
    """
    label_dict = dict()
    stoi = {c:i for i, c in enumerate(ontology)}
    for desc_id, desc in id_map.items():
      doc_id = desc['id'].split('v=')[-1] 
      for punct in PUNCT:
        desc_id = desc_id.replace(punct, '')
      if not desc_id in dur_dict:
        continue

      label_dict[doc_id] = dict()
      dur = dur_dict[desc_id]['duration_second']
      for ann in anno_dict[desc_id+'.mp4']:
        action_class = ontology[ann['Event_Type']] 
        start_sec, end_sec = ann['Temporal_Boundary']  
        start, end = int(start_sec / dur * 100), int(end_sec / dur * 100)
        label_dict[doc_id][(start, end)] = action_class

    return label_dict, stoi
  
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
    candidate_starts = self.candidate_start_ends[idx][:, 0]
    candidate_ends = self.candidate_start_ends[idx][:, 1]
    span_num = len(candidate_starts)
       
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
    labels = [int(self.text_label_dict[self.doc_ids[idx]][(start, end)]) for start, end in zip(origin_candidate_start_ends[:, 0], origin_candidate_start_ends[:, 1])]
    labels = torch.LongTensor(labels)
    labels = fix_embedding_length(labels.unsqueeze(1), self.max_span_num).squeeze(1)
    text_mask = torch.FloatTensor([1. if j < doc_len else 0 for j in range(self.max_token_num)])
    span_mask = continuous_mappings.sum(dim=1)

    # Extract type labels
    type_labels = [self.mention_stoi[self.type_label_dict[self.doc_ids[idx]][(start, end)]] for start, end in zip(origin_candidate_start_ends[:, 0], origin_candidate_start_ends[:, 1])]
    type_labels = torch.LongTensor(type_labels)
    type_labels = fix_embedding_length(type_labels.unsqueeze(1), self.max_span_num).squeeze(1)
    return doc_embeddings,\
        start_mappings,\
        end_mappings,\
        continuous_mappings,\
        width, labels,\
        type_labels,\
        text_mask, span_mask

  def load_video(self, idx):
    '''Load video
    :param idx: int
    :return img_embeddings: FloatTensor of size (batch size, max num. of regions, image embed dim)
    :return mask: LongTensor of size (batch size, max num. of frames)
    '''
    doc_id = self.doc_ids[idx]
    action_embeddings = self.action_embeddings[self.feat_keys[idx]]
    action_embeddings = torch.FloatTensor(action_embeddings)

    if self.use_action_boundary:
      action_segment_embeddings = []
      masks = [] 
      for span_idx, span in enumerate(sorted(self.action_label_dict[doc_id])):
        seg = action_embeddings[span[0]:span[1]+1]
        action_segment_embeddings.append(fix_embedding_length(seg, 30))
        mask = torch.zeros(30, dtype=torch.float)
        mask[:span[1]-span[0]+1] = 1.
        masks.append(mask)
      masks = fix_embedding_length(torch.stack(masks), 20)
      action_segment_embeddings = fix_embedding_length(torch.stack(action_segment_embeddings), 20) 
    else:
      action_segment_embeddings = action_embeddings
      masks = torch.ones(100)

    labels = -1 * np.ones(action_embeddings.size(0))
    for span, label in self.action_label_dict[doc_id].items():
      labels[span[0]:span[1]+1] = self.action_stoi[label]
    labels = torch.LongTensor(labels)

    return action_segment_embeddings, labels, masks

  def __getitem__(self, idx):
    action_embeddings, action_labels, action_mask = self.load_video(idx)
    doc_embeddings,\
    start_mappings,\
    end_mappings,\
    continuous_mappings,\
    width, text_labels,\
    type_labels,\
    text_mask, span_mask = self.load_text(idx)
    return doc_embeddings, start_mappings, end_mappings, continuous_mappings, width, action_embeddings, text_labels, type_labels, action_labels, text_mask, span_mask, action_mask

  def __len__(self):
    return len(self.doc_ids)
