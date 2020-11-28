import numpy as np
import collections
import torch
from torch.utils.data import Dataset
import json
import cv2
from transformers import AutoTokenizer, AutoModel


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
    pad = [torch.zeros(size).unsqueeze(0) for _ in range(L-emb.size(0))]
    emb = torch.cat([emb]+pad, dim=0)
  else:
    emb = emb[:L]
  return emb  

class GroundingDataset(Dataset):
  def __init__(self, doc_json, mention_json, config):
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
         'singleton': boolean, whether the mention is a singleton
    '''
    super(GroundingDataset, self).__init__()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.segment_window = config.get('segment_window', 512) 
    self.max_span_num = config.get('max_span_num', 50)
    self.max_frame_num = config.get('max_frame_num', 20)
    self.max_mention_span = config.get('max_mention_span', 15)

    self.img_dir = config.get('img_dir', './')
    self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    mentions = json.load(codecs.open(mention_json, 'r', 'utf-8'))
    
    # Extract image ids
    self.video_ids = sorted(documents)
    
    # Extract coreference cluster labels
    self.label_dicts = self.create_dict_labels(mentions)
    
    # Extract original mention spans
    self.candidate_start_ends = [np.asarray([[start, end] for start, end in sorted(label_dict[doc_id])]) for doc_id in sorted(documents)]
    
    # Tokenize documents and extract token spans after bert tokenization
    self.tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    self.origin_tokens, self.bert_tokens, self.bert_start_ends = self.tokenize(documents) 
    
    # Extract BERT embeddings
    self.bert_model = AutoModel.from_pretrained(config['bert_model']).to(self.device) 
    self.docs_embeddings = pad_and_read_bert(self.bert_tokens, self.bert_model)

  def tokenize(self, documents):
    '''
    Tokenize the sentences in BERT format. Adapted from https://github.com/ariecattan/coref
    '''
    docs_bert_tokens = []
    docs_origin_tokens = []
    docs_start_end_bert = []

    for doc_id in sorted(documents):
      tokens = documents[doc_id]
      bert_tokens_ids, bert_sentence_ids = [], []
      start_bert_idx, end_bert_idx = [], [] # Start and end token indices for each bert token
      original_tokens = []
      alignment = [] # Store the token id for each character
      bert_cursor = -1
      for i, token in enumerate(tokens):
        sent_id, token_id, token_text, flag_sentence = token
        bert_token = self.tokenizer.encode(token_text, add_special_tokens=True)[1:-1]   
        if bert_token:
          bert_tokens_ids.extend(bert_token)
          bert_start_index = bert_cursor + 1
          start_bert_idx.append(bert_start_index)
          bert_cursor += len(bert_token)
          bert_end_index = bert_cursor
          end_bert_idx.append(bert_end_index)
          original_tokens.append([sent_id, token_id, token_text, flag_sentence])

      docs_bert_tokens.append(bert_token_ids)
      docs_origin_tokens.append(original_tokens)
      start_end = np.concatenate((np.expand_dims(start_bert_idx, 1), np.expand_dims(end_bert_idx, 1)), axis=1)
      docs_start_end_bert.append(start_end)

    return docs_origin_tokens, docs_bert_tokens, docs_start_end_bert 

  def create_dict_labels(self, mentions):
    '''
    :return label_dict: a mapping from doc id to a dict of (start token, end token) -> cluster id 
    '''
    label_dict = collections.defaultdict(dict)
    for m in mentions:
      label_dict[m['doc_id']][(min(m['token_ids']), max(m['tokens_ids']))] = m['cluster_id']
    return label_dict    
  
  def load_text(self, idx):
    '''Load span embeddings for the document
    :param idx: int, doc index
    :return span_embeddings: FloatTensor of size (batch size, max num. spans, span embed dim)
    :return (original_candidate_starts, original_candidate_ends): tuple of LongTensors of size (batch size, max num. spans), start and end of the spans
    :return (bert_candidate_starts, bert_candidate_ends):
    :return span_mask: LongTensor of size (batch size, max num. spans) 
    '''
    # Extract the original spans of the current doc
    origin_candidate_starts = self.candidate_start_ends[idx][0]
    origin_candidate_ends = self.candidate_start_ends[idx][1]

    # Convert the original spans to the bert tokenized spans
    bert_start_ends = self.bert_start_ends[idx]
    bert_candidate_starts = bert_start_ends[origin_candidate_starts]
    bert_candidate_ends = bert_start_ends[origin_candidate_ends]
    span_num = len(bert_candidate_starts)

    # Extract the current doc embedding
    doc_len = len(self.bert_tokens[idx])
    doc_embeddings = self.docs_embeddings[idx][:doc_len]
    start_end_embeddings = torch.cat((doc_embeddings[bert_candidate_starts],
                                      doc_embeddings[bert_candidate_ends]), dim=1)
    continuous_tokens_embeddings, width = get_all_token_embedding(doc_embeddings, 
                                                                   bert_candidate_starts,
                                                                   bert_candidate_ends)
    continuous_tokens_embeddings = torch.FloatTensor([fix_embedding_length(length(emb, self.max_mention_span)\
                                           for emb in continuous_tokens_embeddings])

    # Pad/truncate the outputs to max num. of spans
    start_end_embeddings = fix_embedding_length(start_end_embeddings, self.max_span_num)
    continuous_tokens_embeddings = fix_embedding_length(continuous_tokens_embeddings, self.max_span_num)
    width = fix_embedding_length(width.unsqueeze(1), self.max_span_num).squeeze(1)

    # Extract coreference cluster labels
    labels = [int(self.label_dict[self.video_ids[idx]][(start, end)]) for start, end in zip(origin_candidate_starts, origin_candidate_ends)]

    # TODO Confirm output format
    mask = torch.LongTensor([1. if i < span_num else 0 for i in range(self.max_span_num)])
    return start_end_embeddings, continuous_embeddings, torch.LongTensor(mask), torch.LongTensor(width), torch.LongTensor(labels)

  def load_video(self, filename):
    '''Load video
    :param filename: str, video filename
    :return video_frames: FloatTensor of size (batch size, max num. of frames, width, height, n_channel)
    :return mask: LongTensor of size (batch size, max num. of frames)
    '''    
    # Load video
    cap = cv2.VideoCapture(filename)
    video = []
    while True:
      ret, img = cap.read()
      if not ret:
        print('Number of video frames: {}'.format(len(img)))
        break
      video.append(img)
    
    # Subsample the video frames
    if len(video) < self.max_frame_num:
      video = video.extend([np.zeros((224, 224, 3)) for _ in range(self.max_frame_num-len(video))])
    step = len(video) // self.max_frame_num
    indices = list(range(0, step*self.max_frame_num, step))
     
    images = video[indices] 

    video_frames = []
    for image in images:
      image = Image.fromarray(image)
      # Apply transform to each frame
      if transform is not None:
        image_vec = transform(image)
      video_frames.append(image_vec.unsqueeze(0))

    mask = torch.ones((self.max_frame_num,))
    return torch.cat(video_frames, dim=0), mask


  def __getitem__(self, idx):
    filename = os.path.join(self.img_dir, self.video_ids[idx]+'.mp4')
    video_frames, video_mask = self.load_video(filename)
    start_end_embeddings, continuous_embeddings, span_mask, width, labels = self.load_text(idx)
    return start_end_embeddings, continuous_embeddings, span_mask, width, video_frames, video_mask, labels 
