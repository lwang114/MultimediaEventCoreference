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

def fix_embedding_length(emb, L):
  size = emb.size()[1:]
  if emb.size(0) < L:
    pad = [torch.zeros(size, dtype=emb.dtype).unsqueeze(0) for _ in range(L-emb.size(0))]
    emb = torch.cat([emb]+pad, dim=0)
  else:
    emb = emb[:L]
  return emb  

class VideoM2E2SupervisedCrossmediaDataset(Dataset):
  def __init__(self, config, split='train'):
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
    super(VideoM2E2SupervisedCrossmediaDataset, self).__init__()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.config = config
    self.split = split
    self.max_frame_num = config.get('max_frame_num', 30)

    if config['bert_model'] == 'oneie':
      doc_json = os.path.join(config['data_folder'], f'{split}_oneie.json')
      mention_json = os.path.join(config['data_folder'], f'{split}_oneie_{config["mention_type"]}.json')
    else:
      doc_json = os.path.join(config['data_folder'], f'{split}.json')
      mention_json = os.path.join(config['data_folder'], f'{split}_{config["mention_type"]}.json')    
    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    text_mentions = json.load(codecs.open(mention_json, 'r', 'utf-8'))

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

    # Load action embeddings
    self.action_embeddings = np.load(os.path.join(config['data_folder'], f'{split}_mmaction_feat.npz'))
    
    # Load document embeddings
    if config.bert_model == 'oneie':
      bert_embed_file = f'{doc_json.split(".")[0]}_{config["mention_type"]}.npz'
    else:
      bert_embed_file = '{}_{}.npz'.format(doc_json.split('.')[0], config.bert_model)
    self.docs_embeddings = np.load(bert_embed_file)
    
    self.doc_to_action_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in self.action_embeddings}
    self.doc_to_text_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in self.docs_embeddings}
    
    # Load event info
    self.text_label_dict, self.event_stoi = self.create_text_dict_labels(text_mentions)
    self.action_label_dict, self.action_stoi = self.create_action_dict_labels(id_mapping,\
                                                                              action_anno_dict,\
                                                                              action_dur_dict,\
                                                                              ontology)
    documents = {doc_id:doc for doc_id, doc in documents.items() if doc_id in self.text_label_dict}
    self.documents = documents

    self.data_list = self.create_data_list(self.text_label_dict, self.action_label_dict) # XXX
    print('Number of documents: ', len(documents))
    print('Number of mention-action pairs: ', len(self.data_list))
 
    # Tokenize documents and extract token spans after bert tokenization
    if config.bert_model == 'oneie':
      self.origin_tokens = [documents[k] for k in sorted(documents)]
      self.bert_tokens, self.bert_start_ends, self.clean_start_end_dict = None, None, None
    else:
      self.origin_tokens, self.bert_tokens, self.bert_start_ends, self.clean_start_end_dict = self.tokenize(documents)

  def tokenize(self, documents):
    '''
    Tokenize the sentences in BERT format. Adapted from https://github.com/ariecattan/coref
    '''
    tokenizer = AutoTokenizer.from_pretrained(self.config['bert_model'])
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
            bert_token = tokenizer.encode(token_text, add_special_tokens=True)[1:-1]   
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
    """
    :return text_label_dict: a mapping from doc id to a dict of (start token, end token) -> cluster id 
    :return image_label_dict: a mapping from image id to a dict of (bbox id, x min, y min, x max, y max) -> cluster id 
    """
    stoi = {}
    text_label_dict = {}
    for m in text_mentions:
      if len(m['tokens_ids']) == 0:
        text_label_dict[m['doc_id']][(-1, -1)] = 0
      else:
        start = min(m['tokens_ids'])
        end = max(m['tokens_ids'])
        if not m['doc_id'] in text_label_dict:
          text_label_dict[m['doc_id']] = {}

        if not m['event_type'] in stoi:
          stoi[m['event_type']] = len(stoi)
        text_label_dict[m['doc_id']][(start, end)] = m['event_type'].split('.')[-1] 
    return text_label_dict, stoi
    
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

  def create_data_list(self, 
                       text_label_dict, 
                       action_label_dict):
    """
    :param text_label_dict: {[doc_id]: {[action span]: [action class]}}
    :param action_label_dict: {[doc_id]: {[action span]: [action class]}}
    :returns data_list: list of [(doc_id, [mention span, mention label], [action span, action label])]
    """
    data_list = []
    for doc_idx, doc_id in enumerate(sorted(text_label_dict)):
      for m_idx, m_span in enumerate(sorted(text_label_dict[doc_id])):
        for a_idx, a_span in enumerate(sorted(action_label_dict[doc_id])): 
          data_list.append([doc_idx, doc_id, [m_span, text_label_dict[doc_id][m_span], m_idx], [a_span, action_label_dict[doc_id][a_span], a_idx]])
    return data_list

  def __getitem__(self, idx):
    """ 
    Load mention span embeddings for the document
    
    :param idx: int, doc index
    :return start_end_embeddings: FloatTensor of size (max num. spans, 2, span embed dim)
    :return continuous_tokens_embeddings: FloatTensor of size (max num. spans, max mention span, span embed dim)
    :return mask: FloatTensor of size (max num. spans,)
    :return width: LongTensor of size (max num. spans,)
    :return labels: LongTensor of size (max num. spans,) 
    """
    doc_idx, doc_id, m_info, a_info = self.data_list[idx]
    
    doc_embeddings = self.docs_embeddings[self.doc_to_text_feat[doc_id]]
    doc_embeddings = torch.FloatTensor(doc_embeddings)
    if self.bert_start_ends is not None:
      bert_start_ends = self.bert_start_ends[doc_idx]
      m_start = self.clean_start_end_dict[doc_id][m_info[0][0]]
      m_end = self.clean_start_end_dict[doc_id][m_info[0][1]]
      m_bert_start = bert_start_ends[m_start, 0]
      m_bert_end = bert_start_ends[m_end, 1]
      mention_embedding = doc_embeddings[m_bert_start:m_bert_end+1].mean(dim=0)
    else:
      m_idx = m_info[-1]
      mention_embedding = doc_embeddings[m_idx]

    action_embedding = self.action_embeddings[self.doc_to_action_feat[doc_id]]
    action_embedding = torch.FloatTensor(action_embedding[a_info[0][0]:a_info[0][1]+1])
    action_mask = torch.zeros(self.max_frame_num)
    action_mask[:action_embedding.size(0)] = 1.
    action_embedding = fix_embedding_length(action_embedding, self.max_frame_num)
    
    if not m_info[1] in self.ontology_map:
      coref_label = 0
    elif a_info[1] in self.ontology_map[m_info[1]]:
      coref_label = 1
    else:
      coref_label = 0

    # Extract coreference cluster labels
    return mention_embedding,\
           action_embedding,\
           action_mask,\
           coref_label

  def __len__(self):
    return len(self.data_list)
