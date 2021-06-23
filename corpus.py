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
PLURAL_PRON = ['they', 'them', 'those', 'these', 'we', 'us', 'their', 'our']
SINGULAR = 'singular'
PLURAL = 'plural'
PROPER = 'proper'
NOMINAL = 'nominal'
PRON = 'pronoun'

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

class TextVideoEventDataset(Dataset):
  def __init__(self, config, 
               event_stoi,
               feature_stoi,
               split='train'):
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
    super(TextVideoEventDataset, self).__init__()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.finetune_bert = config.get('finetune_bert', False)
    print(f'Finetune BERT: {self.finetune_bert}')
    self.max_token_num = config.get('max_token_num', 512)
    self.max_span_num = config.get('max_span_num', 80)
    self.max_frame_num = config.get('max_frame_num', 100)
    self.max_mention_span = config.get('max_mention_span', 15)
    self.img_feat_type = config.get('video_feature', 'mmaction_feat')
    self.event_stoi = event_stoi
    feature_stoi['mention_type'] = {NULL:0, PROPER:1, NOMINAL:2, PRON:3}
    feature_stoi['number'] = {NULL:0, SINGULAR:1, PLURAL:2}
    self.feature_stoi = feature_stoi

    doc_json = os.path.join(config['data_folder'], f'{split}.json')
    event_mention_json = os.path.join(config['data_folder'],
                                      f'{split}_events.json')
    entity_mention_json = os.path.join(config['data_folder'],
                                       f'{split}_entities.json')

    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    self.documents = documents
    event_mentions = json.load(codecs.open(event_mention_json, 'r', 'utf-8'))
    entity_mentions = json.load(codecs.open(entity_mention_json, 'r', 'utf-8')) 

    # Load action embeddings
    self.action_embeddings = np.load(os.path.join(config['data_folder'], f'{split}_mmaction_feat.npz'))
       
    # Load document embeddings
    bert_embed_file = '{}_{}.npz'.format(doc_json.split('.')[0], config.bert_model)
    self.docs_embeddings = np.load(bert_embed_file)
    
    # Extract coreference cluster labels
    self.linguistic_feat_types = config.get('linguistic_feature_types', None)
    self.event_label_dict, self.event_feature_dict = self.create_text_dict_labels(event_mentions) 
    self.entity_label_dict, self.entity_feature_dict = self.create_text_dict_labels(entity_mentions)

    self.use_action_boundary = config['use_action_boundary']
    id_mapping_json = os.path.join(config['data_folder'], '../video_m2e2.json')
    action_anno_json = os.path.join(config['data_folder'], '../master.json') # Contain event time stamps
    action_dur_json = os.path.join(config['data_folder'], '../anet_anno.json')
    ontology_json = os.path.join(config['data_folder'], '../ontology.json')
    ontology_map_json = os.path.join(config['data_folder'], '../ontology_mapping.json')
    ie_to_srl_json = os.path.join(config['data_folder'], '../ie_to_srl.json')
    
    id_mapping = json.load(codecs.open(id_mapping_json, 'r', 'utf-8'))
    action_anno_dict = json.load(codecs.open(action_anno_json, 'r', 'utf-8'))
    action_dur_dict = json.load(codecs.open(action_dur_json))

    self.ontology = json.load(codecs.open(ontology_json))
    ontology_map = json.load(codecs.open(ontology_map_json))
    self.ontology_map = {l:k for k, v in ontology_map.items() for l in v}
    self.ie_to_srl_dict = json.load(codecs.open(ie_to_srl_json))

    self.action_label_dict = self.create_action_dict_labels(id_mapping,\
                                                            action_anno_dict,\
                                                            action_dur_dict)

    # Extract doc/image ids
    self.feat_keys = sorted(self.action_embeddings, key=lambda x:int(x.split('_')[-1]))
    if config.debug:
      self.feat_keys = self.feat_keys[:20]
    self.feat_keys = [k for k in self.feat_keys if '_'.join(k.split('_')[:-1]) in self.event_label_dict]
    self.doc_ids = ['_'.join(k.split('_')[:-1]) for k in self.feat_keys]
    documents = {doc_id:documents[doc_id] for doc_id in self.doc_ids}
    print('Number of documents: ', len(self.doc_ids))

    # Tokenize documents and extract token spans after bert tokenization
    self.tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    self.origin_tokens,\
    self.bert_tokens,\
    self.bert_start_ends,\
    clean_start_end_dict = self.tokenize(documents)

    # Extract spans
    self.origin_candidate_start_ends = [np.asarray(sorted(self.event_label_dict[doc_id])) for doc_id in self.doc_ids]
    for doc_id, start_ends in zip(self.doc_ids, self.origin_candidate_start_ends):
      for start, end in start_ends:
        if end >= len(clean_start_end_dict[doc_id]):
          print(doc_id, start, end, len(clean_start_end_dict[doc_id])) 
    self.candidate_start_ends = [np.asarray([[clean_start_end_dict[doc_id][start], clean_start_end_dict[doc_id][end]] for start, end in start_ends])
                                 for doc_id, start_ends in zip(self.doc_ids, self.origin_candidate_start_ends)]

    self.origin_argument_spans = [np.asarray([self.extract_argument(self.event_label_dict[doc_id][span])
                                              for span in sorted(self.event_label_dict[doc_id])]) for doc_id in self.doc_ids]
    self.candidate_argument_spans = [np.asarray([[[clean_start_end_dict[doc_id][start], clean_start_end_dict[doc_id][end]] for start, end in arg_spans] for arg_spans in args_spans])
                                     for doc_id, args_spans in zip(self.doc_ids, self.origin_argument_spans)]

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

  def extract_mention_type(self, pos_tag):
    if pos_tag in ['PRP', 'PRP$']:
      return PRON
    elif pos_tag in ['NNP', 'NNPS']:
      return PROPER
    elif pos_tag in ['NN', 'NNS']:
      return NOMINAL
    else:
      return NOMINAL

  def extract_number(self, token, pos_tag):
    if pos_tag in ['NNPS', 'NNS']:
      return PLURAL 
    elif pos_tag in ['CD']:
      if token in ['one', '1']:
        return SINGULAR
      else:
        return PLURAL
    elif pos_tag in PLURAL_PRON:
      return PLURAL
    else:
      return SINGULAR

  def extract_argument(self, event):
    found = False
    arg_spans = []
    for role in ['arg_0', 'arg_1']:
      span = (-1, -1)
      for rel_idx, arg in enumerate(event['arguments']):
        cur_role = self.ie_to_srl_dict.get(arg['role'], '')
        if cur_role == role:
          span = (arg['start'], arg['end'])
          break
      arg_spans.append(span)
    return np.asarray(arg_spans)

  def create_text_dict_labels(self, text_mentions):
    '''
    :return text_label_dict: a mapping from doc id to a dict of (start token, end token) -> cluster id 
    :return image_label_dict: a mapping from image id to a dict of (bbox id, x min, y min, x max, y max) -> cluster id 
    '''
    text_label_dict = dict()
    ling_feature_dict = dict()
    for m in text_mentions:
      if len(m['tokens_ids']) == 0:
        text_label_dict[m['doc_id']][(-1, -1)] = {'cluster_id': 0,
                                                  'type': 0}
      else:
        start = min(m['tokens_ids'])
        end = max(m['tokens_ids'])
        if not m['doc_id'] in text_label_dict:
          text_label_dict[m['doc_id']] = dict()
          ling_feature_dict[m['doc_id']] = dict()
        
        if 'event_type' in m:
          mention_class = m['event_type']
        else:
          mention_class = m['entity_type']

        text_label_dict[m['doc_id']][(start, end)] = {'cluster_id': m['cluster_id'],
                                                      'type': mention_class,
                                                      'arguments': m.get('arguments', [])}
        
        ling_feature_dict[m['doc_id']][(start, end)] = dict() 
        for feat_type in self.linguistic_feat_types:
          if feat_type == 'mention_type':
            ling_feature_dict[m['doc_id']][(start, end)][feat_type] = self.extract_mention_type(m['pos_tag'])
          elif feat_type == 'number':
            ling_feature_dict[m['doc_id']][(start, end)][feat_type] = self.extract_number(m['tokens'], m['pos_tag'])
          else:
            ling_feature_dict[m['doc_id']][(start, end)][feat_type] = m[feat_type]
    return text_label_dict, ling_feature_dict
  
  def create_action_dict_labels(self, 
                               id_map,
                               anno_dict,
                               dur_dict):
    """
    :param id2desc: {[youtube id]: [description id with puncts]} 
    :param anno_dict: {[description id]: list of {'Temporal_Boundary': [float, float], 'Event_Type': int}}  
    :param dur_dict: {[description id]: {'duration_second': float}}
    :param ontology: list of mention classes
    :returns image_label_dict: {[doc_id]: {[action span]: [action class]}}
    """
    label_dict = dict()
    for desc_id, desc in id_map.items():
      doc_id = desc['id'].split('v=')[-1] 
      for punct in PUNCT:
        desc_id = desc_id.replace(punct, '')
      if not desc_id in dur_dict:
        continue

      label_dict[doc_id] = dict()
      dur = dur_dict[desc_id]['duration_second']
      for ann in anno_dict[desc_id+'.mp4']:
        action_class = self.ontology['event'][ann['Event_Type']] 
        start_sec, end_sec = ann['Temporal_Boundary']  
        start, end = int(start_sec / dur * 100), int(end_sec / dur * 100)
        label_dict[doc_id][(start, end)] = action_class

    return label_dict
  
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
    bert_tokens = self.bert_tokens[idx]
    doc_len = len(bert_tokens)
    if self.finetune_bert = True:
      doc_embeddings = fix_embedding_length(bert_tokens.unsqueeze(-1), self.max_token_num).squeeze(-1)
    else:
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

    # Convert the argument spans to the bert tokenized spans
    if not len(self.origin_argument_spans[idx].shape):
      n_args = 2
      arg_starts = []
      arg_ends = []
      start_arg_mappings = torch.zeros(n_args*self.max_span_num, self.max_token_num)
      end_arg_mappings = torch.zeros(n_args*self.max_span_num, self.max_token_num)
      continuous_arg_mappings = torch.zeros(n_args*self.max_span_num,
                                            self.max_mention_span,
                                            self.max_token_num)
      arg_width = torch.zeros(n_args*self.max_span_num)
    else:
      n_args = self.candidate_argument_spans[idx].shape[1]
      arg_starts = self.candidate_argument_spans[idx][:, :, 0].flatten()
      arg_ends = self.candidate_argument_spans[idx][:, :, 1].flatten()
      bert_arg_starts = bert_start_ends[arg_starts, 0]
      bert_arg_ends = bert_start_ends[arg_ends, 1] 
      start_arg_mappings, end_arg_mappings, continuous_arg_mappings, arg_width =\
        get_all_token_mapping(bert_arg_starts,
                              bert_arg_ends,
                              self.max_token_num,
                              self.max_mention_span)
      start_arg_mappings = fix_embedding_length(start_arg_mappings, n_args*self.max_span_num)
      end_arg_mappings = fix_embedding_length(end_arg_mappings, n_args*self.max_span_num)
      continuous_arg_mappings = fix_embedding_length(continuous_arg_mappings, n_args*self.max_span_num)
      arg_width = fix_embedding_length(arg_width.unsqueeze(1), n_args*self.max_span_num).squeeze(1)

    # Pad/truncate the outputs to max num. of spans
    start_mappings = fix_embedding_length(start_mappings, self.max_span_num)
    end_mappings = fix_embedding_length(end_mappings, self.max_span_num)
    continuous_mappings = fix_embedding_length(continuous_mappings, self.max_span_num)
    width = fix_embedding_length(width.unsqueeze(1), self.max_span_num).squeeze(1)

    # Extract coreference cluster labels
    labels = [int(self.event_label_dict[self.doc_ids[idx]][(start, end)]['cluster_id'])\
              for start, end in zip(origin_candidate_start_ends[:, 0], origin_candidate_start_ends[:, 1])]
    labels = torch.LongTensor(labels)
    labels = fix_embedding_length(labels.unsqueeze(1), self.max_span_num).squeeze(1)
    event_labels = [self.event_stoi[self.event_label_dict[self.doc_ids[idx]][(start, end)]['type']]\
                    for start, end in zip(origin_candidate_start_ends[:, 0], origin_candidate_start_ends[:, 1])]
    event_labels = torch.LongTensor(event_labels)
    event_labels = fix_embedding_length(event_labels.unsqueeze(1), self.max_span_num).squeeze(1)

    text_mask = torch.FloatTensor([1. if j < doc_len else 0 for j in range(self.max_token_num)])
    span_mask = continuous_mappings.sum(dim=1)

    # Extract linguistic feature labels
    linguistic_labels = {k:[] for k in self.linguistic_feat_types}
    for start, end in zip(origin_candidate_start_ends[:, 0], origin_candidate_start_ends[:, 1]):
      feat_dict = self.event_feature_dict[self.doc_ids[idx]][(start, end)]
      for k in self.linguistic_feat_types:
        linguistic_labels[k].append(self.feature_stoi[k][feat_dict[k]])

    for k in linguistic_labels:  
      linguistic_labels[k] = torch.LongTensor(linguistic_labels[k])
      linguistic_labels[k] = fix_embedding_length(
                               linguistic_labels[k].unsqueeze(1), 
                               self.max_span_num).squeeze(1)
 
    arg_linguistic_labels = {k:[] for k in self.linguistic_feat_types}
    for start, end in zip(arg_starts, arg_ends):
      feat_dict = self.entity_feature_dict[self.doc_ids[idx]].get((start, end), dict())
      for k in self.linguistc_feat_types:
        arg_linguistic_labels[k].append(self.feature_stoi[k][feat_dict.get(k, NULL)]) 

    for k in arg_linguistic_labels:  
      arg_linguistic_labels[k] = torch.LongTensor(arg_linguistic_labels[k])
      arg_linguistic_labels[k] = fix_embedding_length(
                                   arg_linguistic_labels[k].unsqueeze(1), 
                                   n_args*self.max_span_num).squeeze(1)
   
    return {'doc_embeddings': doc_embeddings,
            'start_mappings': start_mappings,
            'end_mappings': end_mappings,
            'continuous_mappings': continuous_mappings,
            'width': width,
            'start_arg_mappings': start_arg_mappings,
            'end_arg_mappings': end_arg_mappings,
            'continuous_arg_mappings': continuous_arg_mappings,
            'arg_width': arg_width,
            'cluster_labels': labels,
            'event_labels': event_labels,
            'linguistic_labels': linguistic_labels,
            'arg_linguistic_labels': arg_linguistic_labels,
            'text_mask': text_mask,
            'span_mask': span_mask,
            'n_args': n_args}

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
      if label in self.ontology_map:
        labels[span[0]:span[1]+1] = self.event_stoi[self.ontology_map[label]]
    labels = torch.LongTensor(labels)

    return {'action_embeddings': action_segment_embeddings, 
            'action_labels': labels, 
            'action_mask': masks}

  def __getitem__(self, idx):
    inputs = self.load_video(idx)
    inputs.update(self.load_text(idx))
    return inputs

  def __len__(self):
    return len(self.doc_ids)
