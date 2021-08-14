import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import os
import json
import numpy as np
from tqdm import tqdm
import codecs
import collections

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
    length = torch.LongTensor(length)
    return start_mappings, end_mappings, span_mappings, length

def fix_embedding_length(emb, L, padding=0):
  size = emb.size()[1:]
  if emb.size(0) < L:
    if padding == 0:
        pad = torch.zeros((L-emb.size(0), *size), dtype=emb.dtype)
    else:
        pad = padding * torch.ones((L-emb.size(0), *size), dtype=emb.dtype)
    emb = torch.cat([emb, pad], dim=0)
  else:
    emb = emb[:L]
  return emb

def pad_and_read_bert(bert_token_ids, bert_model):
  length = np.array([len(d) for d in bert_token_ids])
  max_length = max(length)

  if max_length > 512:
      raise ValueError('Error! Segment too long!')

  device = bert_model.device
  docs = torch.tensor([doc + [0] * (max_length - len(doc)) for doc in bert_token_ids], device=device)
  attention_masks = torch.tensor([[1] * len(doc) + [0] * (max_length - len(doc)) for doc in bert_token_ids], device=device)
  with torch.no_grad():
      embeddings, _ = bert_model(docs, attention_masks)

  return embeddings, length


class VisProDataset:
  def __init__(self, config,
               split='train',
               class_stoi=None):
    """
    Dataloader for the VisPro dataset: https://github.com/HKUST-KnowComp/Visual_PCR/blob/master/README.md

    Input meta info fields:
      "pronoun_info": a list of dicts of
          "current_pronoun": [int, int] (start and end index, inclusive),
          "candidate_NPs": list of [int, int]'s,
          "correct_NPs": list of [int, int]'s,
          "reference_type": int,
          "not_discussed": bool,
      "sentences": list of list of strs,
      "image_file": str, 
      "clusters": list of list of [int, int],
      "object_detection": list of ints, 
      "doc_key": str,
      "speakers": ??
    """
    super(VisProDataset, self).__init__()
    self.debug = config['debug']
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.max_token_num = config.get('max_token_num', 512)
    self.max_span_num = config.get('max_span_num', 100)
    self.max_pair_num = int(self.max_span_num * (self.max_span_num - 1) // 2)
    self.max_object_num = config.get('max_object_num', 20)
    self.max_frame_num = config.get('max_frame_num', 100)
    self.max_mention_span = config.get('max_mention_span', 15)
    self.add_glove = config.get('add_glove', False)
    
    mention_json = os.path.join(config['data_folder'], f'{split}.vispro.1.1.jsonlines')
    self.tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    self.mention_dict,\
    self.visual_dict,\
    self.documents = self.extract_meta_info(mention_json)
    self.doc_ids = sorted(self.documents)
    print(f'Number of documents in {split}: {len(self.doc_ids)}')
    
    self.origin_tokens,\
    self.bert_tokens,\
    self.bert_start_ends,\
    self.alignments = self.tokenize(self.documents)
    self.candidate_start_ends = [np.asarray(sorted(self.mention_dict[doc_id])) for doc_id in sorted(self.mention_dict)]
  
    bert_embed_file = os.path.join(config['data_folder'], f'{split}_{config["bert_model"]}.npz')
    glove_embed_file = os.path.join(config['data_folder'], f'{split}_glove_embeddings.npz')
    elmo_embed_file = os.path.join(config['data_folder'], f'ELMo_Vispro.npz')

    if not os.path.exists(bert_embed_file):
      self.extract_bert_embeddings(config['bert_model'], bert_embed_file)
    if not os.path.exists(glove_embed_file):
      glove_file = config.get('glove_file', 'm2e2/data/glove/glove.840B.300d.txt')
      self.extract_glove_embeddings(glove_file, glove_embed_file)

    self.embed_type = config['embed_type']
    self.docs_embeddings = np.load(bert_embed_file)
    self.glove_embeddings = np.load(glove_embed_file)
    if not os.path.exists(elmo_embed_file):
      self.elmo_embeddings = None
    else:
      self.elmo_embeddings = np.load(elmo_embed_file)

    if not class_stoi:
      self.class_stoi = {str(i):i for i in range(100)}
    else:
      self.class_stoi = class_stoi

  def extract_meta_info(self, mention_json):
    """
    Returns:
      mention_dict: mapping from "doc_id" (concatenation of doc_key and image_file) to
        a mapping from span (int, int) to int cluster id
    """
    mention_dict = dict()
    visual_dict = dict()
    documents = dict()
    for line in open(mention_json, 'r'):
      if self.debug:
          if len(mention_dict) >= 20:
              break
      doc_dict = json.loads(line)
      
      doc_id = doc_dict['doc_key']
      clusters = doc_dict['clusters']
      if len(clusters):
          mention_dict[doc_id] = {tuple(span):{'cluster_id':c_idx+1,
                                               'candidate_NPs':[],
                                               'class':0}
                                  for c_idx, c in enumerate(clusters) for span in c}
      else:
          mention_dict[doc_id] = {(-1, -1):{'cluster_id':0,
                                            'candidate_NPs':[],
                                            'class':0}}

      for p in doc_dict['pronoun_info']:
          span = tuple(p['current_pronoun'])
          if not span in mention_dict[doc_id]: # Ignore non-referential pronouns
              continue
          mention_dict[doc_id][span]['class'] = p.get('class', 0)
          mention_dict[doc_id][span]['candidate_NPs'] = p['candidate_NPs']

          for m in p['candidate_NPs']:
              m = tuple(m)
              if not m in mention_dict[doc_id]: # Add singletons
                  mention_dict[doc_id][m] = {'cluster_id':0,
                                             'candidate_NPs':[],
                                             'class':0}

      visual_dict[doc_id] = [] 
      for c in doc_dict['object_detection']:
          visual_dict[doc_id].append(c)
      
      documents[doc_id] = [[sent_id, 0, token, 0]\
                            for sent_id, sent in enumerate(doc_dict['sentences'])\
                              for token in sent]
    return mention_dict, visual_dict, documents
  
  def tokenize(self, documents):
    bert_tokens_all = []
    bert_start_ends_all = []
    origin_tokens_all = []
    alignments = []
    
    for doc_id in sorted(documents):
      tokens = documents[doc_id]
      bert_tokens = []
      start_bert_idx, end_bert_idx = [], []
      origin_tokens = []
      alignment = []
      bert_cursor = -1
      for i, token in enumerate(tokens):
        sent_id, token_id, token_text, flag_sentence = token
        bert_token = self.tokenizer.encode(token_text, add_special_tokens=True)[1:-1]
        if bert_token:
          bert_start_index = bert_cursor + 1
          bert_tokens.extend(bert_token)
          start_bert_idx.append(bert_start_index)
          bert_cursor += len(bert_token)
          bert_end_index = bert_cursor
          end_bert_idx.append(bert_end_index)
          alignment.extend([i]*len(bert_token))
        else:
          start_bert_idx.append(-1)
          end_bert_idx.append(-1)
          
        origin_tokens.append(token)
      bert_tokens_all.append(bert_tokens)
      origin_tokens_all.append(origin_tokens)
      alignments.append(alignment)
      start_end = np.concatenate((np.expand_dims(start_bert_idx, 1), np.expand_dims(end_bert_idx, 1)), axis=1)
      bert_start_ends_all.append(start_end)
    return origin_tokens_all, bert_tokens_all, bert_start_ends_all, alignments

  def split_doc_into_segments(self, bert_tokens, sentence_ids):
    """
    Args :
        bert_tokens : list of ints,
        sentence_ids : list of ints,
    
    Returns :
        segments : list of ints, the start position of each segment of the document 
    """
    segments = [0]
    current_token = 0
    max_segment_length = 510 # 512-2 to account for special tokens
    while current_token < len(bert_tokens):
      end_token = min(len(bert_tokens) - 1, current_token + max_segment_length - 1)
      sentence_end = sentence_ids[end_token]
      if end_token != len(bert_tokens) - 1 and sentence_ids[end_token + 1] == sentence_end:
        while end_token >= current_token and sentence_ids[end_token] == sentence_end:
          end_token -= 1

        if end_token < current_token:
          raise ValueError(bert_tokens)

      current_token = end_token + 1
      segments.append(current_token)

    return segments   
        
  def extract_bert_embeddings(self, bert_type, out_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_model = AutoModel.from_pretrained(bert_type).to(device)
    docs_embs = dict()

    with torch.no_grad():
      bert_tokens_segments = []
      doc_ids_segments = []
      for doc_id, tokens, origin_tokens, alignment in zip(self.doc_ids, self.bert_tokens, self.origin_tokens, self.alignments):
        sentence_ids = [origin_tokens[token_idx][0] for token_idx in alignment]

        # Segment the tokens to chunks of at most 512 tokens
        segments = self.split_doc_into_segments(tokens, sentence_ids)
        bert_segments, start_end_segment = [], []
        delta = 0

        # Extract spans for each segment
        for start, end in zip(segments, segments[1:]):
          bert_ids = tokens[start:end]
          bert_segments.append(bert_ids)

        bert_tokens_segments.extend(bert_segments)
        doc_segment = [doc_id] * (len(segments) - 1)
        doc_ids_segments.extend(doc_segment)
   
      bert_embeddings = dict()
      total = len(doc_ids_segments)
      n_batch = total // 8 + 1 if total % 8 != 0 else total // 8
      for b in tqdm(range(n_batch)):
        start_idx = b * 8
        end_idx = min((b + 1) * 8, total)
        batch_idxs = list(range(start_idx, end_idx))
        doc_ids_batch = [doc_ids_segments[i] for i in batch_idxs] 
        bert_tokens_batch = [bert_tokens_segments[i] for i in batch_idxs]
        bert_embeddings_batch, docs_length = pad_and_read_bert(bert_tokens_batch, bert_model)
        docs_length2 = [len(ts) for ts in bert_tokens_batch]

        for idx, doc_id in enumerate(doc_ids_batch):
          bert_embedding = bert_embeddings_batch[idx][:docs_length[idx]].cpu().detach().numpy() 
          if not doc_id in bert_embeddings:
              bert_embeddings[doc_id] = bert_embedding
          else:
              bert_embeddings[doc_id] = np.concatenate([bert_embeddings[doc_id], bert_embedding], axis=-2)
      np.savez(out_file, **bert_embeddings) 

  def extract_glove_embeddings(self, glove_file, out_file):
    vocab = {'$$$UNK$$$': 0}
    dimension = 300
    # Compute vocab of the documents
    for doc_id in sorted(self.documents):
        tokens = self.documents[doc_id]
        for token in tokens:
            if not token[2].lower() in vocab:
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
                embed= [float(x) for x in segments[-300:]]
                embed_matrix.append(embed)
                vocab_emb[word] = len(vocab_emb)
    print('Vocabulary size with embeddings: {}'.format(len(vocab_emb)))   

    # Convert the documents into embedding sequence
    doc_embeddings = {}
    for idx, doc_id in enumerate(sorted(self.documents)):
        tokens = self.documents[doc_id]
        doc_embedding = []
        for token in tokens:
            token_id = vocab_emb.get(token[2].lower(), 0)
            doc_embedding.append(embed_matrix[token_id])
        doc_embeddings[doc_id] = np.asarray(doc_embedding)
    np.savez(out_file, **doc_embeddings)

  def align_glove_with_bert(self, glove_embs, alignment):
    token_num = len(alignment)
    glove_dim = glove_embs.shape[-1]
    aligned_glove_embs = np.zeros((token_num, glove_dim))
    for i, token_idx in enumerate(alignment):
        aligned_glove_embs[i] = glove_embs[token_idx]
    return aligned_glove_embs

  def load_bert(self, idx):
    doc_id = self.doc_ids[idx]
    alignment = self.alignments[idx]
    candidate_starts = self.candidate_start_ends[idx][:, 0]
    candidate_ends = self.candidate_start_ends[idx][:, 1]
    span_num = len(candidate_starts)
    
    # Extract the current doc embedding
    bert_tokens = self.bert_tokens[idx]
    doc_len = len(bert_tokens)

    doc_embeddings = self.docs_embeddings[doc_id][:doc_len]
    if self.add_glove:
      glove_embeddings = self.align_glove_with_bert(self.glove_embeddings[doc_id], alignment)
      doc_embeddings = np.concatenate([doc_embeddings, glove_embeddings], axis=-1)
    
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
    return doc_embeddings,\
           start_mappings,\
           end_mappings,\
           continuous_mappings,\
           width

  def load_elmo(self, idx):
    doc_id = self.doc_ids[idx]
    tokens = self.documents[doc_id]
    caption = [t for t in tokens if t[0] == 0]
    caption_len = len(caption)
    caption_embs = torch.zeros((caption_len, 3072))

    caption_mentions = [span for span in self.candidate_start_ends[idx] if tokens[span[0]][0] == 0]  
    caption_mention_embs = self.elmo_embeddings[f'{doc_id}:cap'] 
    mention_lens = self.elmo_embeddings[f'{doc_id}:cap:lengths'] # TODO confirm
    print('len(caption_mentions), len(caption_mention_lens), expected equal: ', len(caption_mentions), len(caption_mention_lens)) # XXX
    assert len(caption_mentions) == len(caption_mention_lens)
    offset = 0
    for m_idx, (start, end) in caption_mentions:
      caption_embs[start:end+1] = caption_mention_embs[offset:offset+mention_lens[m_idx]].reshape(-1, 3072)
      offset += len(mention_lens[m_idx])

    dialogue_embs = self.elmo_embeddings[doc_id]
    start_mappings, end_mappings, continuous_mappings, width =\
      get_all_token_mapping(self.candidate_start_ends[idx, 0],
                            self.candidate_start_ends[idx, 1],
                            self.max_token_num,
                            self.max_mention_span) 
    width = torch.LongTensor([min(w, self.max_mention_span) for w in width])
    
    doc_embeddings = torch.cat([caption_embs, dialogue_embs], dim=0)
    return doc_embeddings,
           start_mappings,\
           end_mappings,\
           continuous_mappings,\
           width

  def load_text(self, idx):
    if self.embed_type == 'bert':
      doc_embeddings,\
      start_mappings,\
      end_mappings,\
      continuous_mappings,\
      width = self.load_bert(idx)
    elif self.embed_type == 'elmo':
      doc_embeddings,\
      start_mappings,\
      end_mappings,\
      continuous_mappings,\
      width = self.load_elmo(idx)

    # Pad/truncate the outputs to max num. of spans
    start_mappings = fix_embedding_length(start_mappings, self.max_span_num)
    end_mappings = fix_embedding_length(end_mappings, self.max_span_num)
    continuous_mappings = fix_embedding_length(continuous_mappings, self.max_span_num)
    width = fix_embedding_length(width.unsqueeze(1), self.max_span_num).squeeze(1)

    # Extract coreference cluster labels
    labels = [self.mention_dict[doc_id][(start, end)]['cluster_id']\
              for start, end in zip(candidate_starts, candidate_ends)]
    type_labels = [self.mention_dict[doc_id][(start, end)]['class']\
                   for start, end in zip(candidate_starts, candidate_ends)]
    span_map = {s:i for i, s in enumerate(sorted(self.mention_dict[doc_id]))}
    first_idxs = [first_idx for first_idx, span in enumerate(sorted(self.mention_dict[doc_id]))\
                  for _ in self.mention_dict[doc_id][span]['candidate_NPs']]
    second_idxs = [span_map[tuple(second_span)] for _, span in enumerate(sorted(self.mention_dict[doc_id]))\
                   for second_span in self.mention_dict[doc_id][span]['candidate_NPs']]
    pairwise_labels = [int((labels[first_idx] == labels[second_idx]) & (labels[first_idx] != 0) & (labels[second_idx] != 0))
                       for first_idx, second_idx in zip(first_idxs, second_idxs)]

    labels = torch.LongTensor(labels)
    labels = fix_embedding_length(labels.unsqueeze(1), self.max_span_num).squeeze(1)

    type_labels = torch.LongTensor(type_labels)
    type_labels = fix_embedding_length(type_labels.unsqueeze(1), self.max_span_num).squeeze(1)
    
    first_idxs = torch.LongTensor(first_idxs)
    first_idxs = fix_embedding_length(first_idxs.unsqueeze(1),
                                      self.max_pair_num,
                                      padding=-1).squeeze(1)
    second_idxs = torch.LongTensor(second_idxs)
    second_idxs = fix_embedding_length(second_idxs.unsqueeze(1),
                                       self.max_pair_num,
                                       padding=-1).squeeze(1)
    pairwise_labels = torch.LongTensor(pairwise_labels)
    pairwise_labels = fix_embedding_length(pairwise_labels.unsqueeze(1), self.max_pair_num).squeeze(1)
    
    text_mask = torch.FloatTensor([1. if j < doc_len else 0 for j in range(self.max_token_num)])
    span_mask = continuous_mappings.sum(dim=1)

    return {'doc_embeddings': doc_embeddings,
            'start_mappings': start_mappings,
            'end_mappings': end_mappings,
            'continuous_mappings': continuous_mappings,
            'width': width,
            'cluster_labels': labels,
            'first_idxs': first_idxs,
            'second_idxs': second_idxs,
            'pairwise_labels': pairwise_labels,
            'type_labels': type_labels,
            'text_mask': text_mask,
            'span_mask': span_mask}
 
  def load_image(self, idx):
    doc_id = self.doc_ids[idx]
    class_labels = torch.LongTensor([0]+self.visual_dict[doc_id])
    object_num = class_labels.size(0)
    class_mask = torch.zeros(self.max_object_num)
    class_mask[:object_num] = 1.
    class_labels = fix_embedding_length(class_labels.unsqueeze(1), self.max_object_num).squeeze(1) 
    
    return {'visual_embeddings': torch.zeros((self.max_object_num, 400)), # XXX to make it consistent with video m2e2
            'class_labels': class_labels,
            'class_mask': class_mask}

  def __getitem__(self, idx):
    inputs = self.load_image(idx)
    inputs.update(self.load_text(idx))
    return inputs
  
  def __len__(self):
    return len(self.doc_ids)  
