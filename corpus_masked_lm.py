import numpy as np
import json
import os
import codecs
from copy import deepcopy
import collections
import torchvision.transforms as transforms
from transformers import AutoTokenizer

def fix_embedding_length(emb, L, pad_val=0):
  size = emb.size()[1:]
  if emb.size(0) < L:
    if pad_val == 0:
      pad = torch.zeros((L-emb.size(0), *size), dtype=emb.dtype).unsqueeze(0)
    else:
      pad = pad_val*torch.ones((L-emb.size(0), *size), dtype=emb.dtype).unsqueeze(0)
    emb = torch.cat((emb, pad), dim=0)
  else:
    emb = emb[:L]
  return emb

def get_mention_mapping(spans, 
                        max_token_num, 
                        max_mention_span,
                        bos_token=False):
  try:
    span_num = len(spans)
  except:
    raise ValueError('Invalid type for spans={}'.format(spans))
  offset = 1 if bos_token else 0

  start_mappings = torch.zeros((span_num+offset, 
                                max_token_num+offset), dtype=torch.float) 
  end_mappings = torch.zeros((span_num+offset,
                              max_token_num+offset), dtype=torch.float)
  continuous_mappings = torch.zeros((span_num+offset,
                                     max_mention_span,
                                     max_token_num), dtype=torch.float)
  length = torch.zeros(span_num+offset, dtype=torch.long)
  if bos_token:
    start_mappings[0, 0] = 1.
    end_mappings[0, 0] = 1.
    continuous_mappings[0, 0, 0] = 1.
     
  for span_idx, (b, e) in enumerate(spans):
    if e > max_token_num:
      continue
    start_mappings[span_idx, b+1] = 1.
    end_mappings[span_idx, e+1] = 1. 
    for i, x in enumerate(range(b, e+1)):
      continuous_mappings[span_idx, i, x] = 1.
    length[span_idx] = e-b+1

  return start_mappings, end_mappings, continuous_mappings, length

class TextEventMLMDataset(Dataset):
  def __init__(self, config,
               feature_stoi,               
               split):
    super(TextEventMLMDataset, self).__init__()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.tokenizer = AutoTokenizer.from_pretrained(config['bert_model']) 
    
    doc_json = os.path.join(config['data_folder'], f'{split}.json')
    event_mention_json = os.path.join(config['data_loader'],
                                      f'{split}_events.json')
    self.linguistic_feat_types = config.get('linguistic_feature_types', None)

    self.documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    self.doc_ids = sorted(self.documents)
    if config.debug:
      self.doc_ids = self.doc_ids[:20]

    event_mentions = json.load(codecs.open(event_mention_json, 'r', 'utf-8'))
    self.event_label_dict = self.create_text_dict_labels(event_mentions)
  
    self.event_spans = [np.asarray(sorted(self.event_label_dict[doc_id]))\
                          for doc_id in self.doc_ids]

  def create_text_dict_labels(self, text_mentions):
    text_label_dict = dict()
    for m in text_mentions:
      if len(m['tokens_ids']) == 0:
        text_label_dict[m['doc_id']][(-1, -1)] = {'cluster_id': 0}
      else:
        start = min(m['tokens_ids'])
        end = max(m['tokens_ids'])
        if not m['doc_id'] in text_label_dict:
          start = min(m['tokens_ids'])
          end = max(m['tokens_ids'])
          text_label_dict[m['doc_id']][(start, end)] = {'cluster_id': m['cluster_id']}
          for feat_type in self.linguistic_feat_types:
            if feat_type == 'mention_type':
              text_label_dict[m['doc_id']][(start, end)][feat_type] = self.extract_mention_type(m[feat_type])
            elif feat_type == 'number': 
              text_label_dict[m['doc_id']][(start, end)][feat_type] = self.extract_number(m[feat_type])
            else:
              text_label_dict[m['doc_id']][(start, end)][feat_type] = m[feat_type]
    return text_label_dict 

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

  ''' TODO
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
  '''

  def is_feature_consistent(self, feat1, feat2, feat_type):
    if feat_type in ['number', 'event_type', 'entity_type']:
      return (feat1 == feat2)
    else:
      return True

  def mask_event(self, tokens, 
                 spans, 
                 linguistic_labels):
    """ Mask tokens for each event mention """
    doc_len = len(tokens)
    num_spans = len(spans)

    tokens_list = []
    for i in range(num_spans):
      x = deepcopy(tokens)
      for start, end in spans[i]:
        for mask_idx in range(start, doc_len):
          x[mask_idx] = self.tokenizer.mask_token  

      for feat_type in self.linguistic_features:      
        for j in range(1, i+1):
          if not is_feature_consistent(linguistic_labels[feat_type][i],
                                       linguistic_labels[feat_type][j-1], 
                                       feat_type):
            for start, end in spans[j]:
              for mask_idx in range(start, end+1):
                x[mask_idx] = self.tokenizer.mask_token
      tokens_list.append(x)
    return tokens_list

  def __item__(self, idx):
    doc_id = self.doc_ids[idx]
    tokens = self.documents[doc_id]
    encodings = self.tokenizer([self.tokenizer.bos_token]+tokens, 
                               padding=True,
                               truncation=True,
                               max_length=self.max_token_num,
                               return_attention_mask=True)
    input_ids = encodings['input_ids'].squeeze(0)
    text_mask = encodings['attention_mask'].squeeze(0) 
    event_spans = self.event_spans[idx]

    # Extract labels
    label_dict = self.event_label_dict[doc_id]
    cluster_labels = [label_dict[tuple(span.tolist())]['cluster_id'] for span in event_spans]
    cluster_labels = torch.LongTensor(cluster_labels)
    cluster_labels = fix_embedding_length(cluster_labels.unsqueeze(-1), self.max_span_num, pad_val=-1).squeeze(-1)

    linguistic_labels = {k:[] for k in self.linguistic_feat_types}
    for start, end in zip(event_spans): 
      feat_dict = self.event_label_dict[doc_id][(start, end)]
      for k in self.linguistic_feat_types:
        linguistic_labels[k].append(self.feature_stoi[k][feat_dict[k]])

    # Add mask tokens
    event_masked_tokens = self.mask_event(tokens,
                                          event_spans,
                                          linguistic_labels)

    event_mask_input_ids = self.tokenizer(
                             [[self.tokenizer.bos_token]+sent\
                                 for sent in event_masked_tokens],
                             padding=True,
                             truncation=True,
                             max_length=self.max_token_num,
                             return_attention_mask=True)['input_ids']
    event_mask_input_ids = fix_embedding_length(event_mask_input_ids, self.max_span_num)

    # Extract mention mappings
    start_mappings,\
    end_mappings,\
    span_mappings,\
    length = get_mention_mapping(event_spans, 
                                 self.max_token_num,
                                 self.max_mention_span,
                                 bos_token=True)

    # TODO Argument masking
    return {'input_ids': input_ids,
            'event_mask_input_ids': event_mask_input_ids,
            'start_mappings': start_mappings,
            'end_mappings': end_mappings,
            'continuous_mappings': continuous_mappings,
            'width': width,
            'cluster_labels': cluster_labels,
            'linguistic_labels': linguistic_labels,
            'text_mask': text_mask}
  
  def __len__(self): 
    return len(self.doc_ids)
