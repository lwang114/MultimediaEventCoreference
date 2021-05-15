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

PUNCT = [',', '.', '\'', '\"', ':', ';', '?', '!', '<', '>', '~', '%', '$', '|', '/', '@', '#', '^', '*']
def fix_embedding_length(emb, L):
  size = emb.size()[1:]
  if emb.size(0) < L:
    pad = [torch.zeros(size, dtype=emb.dtype, device=emb.device).unsqueeze(0) for _ in range(L-emb.size(0))]
    emb = torch.cat([emb]+pad, dim=0)
  else:
    emb = emb[:L]
  return emb  

class VideoM2E2ActionDataset(Dataset):
  def __init__(self, config, split='train'):
    super(VideoM2E2ActionDataset, self).__init__()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.config = config
    self.split = split
    self.max_frame_num = config.get('max_frame_num', 100)
    self.visual_feat_type = config['visual_feature_type']

    doc_json = os.path.join(config['data_folder'], f'{split}.json')
    id_mapping_json = os.path.join(config['data_folder'], '../video_m2e2.json')
    action_anno_json = os.path.join(config['data_folder'], '../master.json')
    action_dur_json = os.path.join(config['data_folder'], '../anet_anno.json')
    ontology_json = os.path.join(config['data_folder'], '../ontology.json')

    documents = json.load(codecs.open(doc_json, 'r', 'utf-8')) 
    id_mapping = json.load(codecs.open(id_mapping_json, 'r', 'utf-8'))
    action_anno_dict = json.load(codecs.open(action_anno_json, 'r', 'utf-8'))
    action_dur_dict = json.load(codecs.open(action_dur_json, 'r', 'utf-8'))

    ontology_dict = json.load(codecs.open(ontology_json))
    self.ontology = ontology_dict['event']

    # Load action embeddings
    self.action_embeddings = np.load(os.path.join(config['data_folder'], f'{split}_{self.visual_feat_type}.npz'))
    self.doc_to_feat = {'_'.join(feat_id.split('_')[:-1]):feat_id for feat_id in self.action_embeddings}
    frame_dict = {'_'.join(feat_id.split('_')[:-1]):self.action_embeddings[feat_id].shape[0] for feat_id in self.action_embeddings}

    # Load action labels
    label_dict = self.create_dict_labels(id_mapping, action_anno_dict, action_dur_dict, frame_dict)
    self.label_dict = {doc_id:label_dict[doc_id] for doc_id in documents}

    self.data_list = self.create_data_list(self.label_dict)
    print('Number of documents: ', len(self.label_dict))
    print('Number of action segments: ', len(self.data_list))

  def create_dict_labels(self,
                         id_map,
                         anno_dict,
                         dur_dict,
                         frame_dict):
    label_dict = dict()
    for desc_id, desc in id_map.items():
      doc_id = desc['id'].split('v=')[-1]
      for punct in PUNCT:
        desc_id = desc_id.replace(punct, '')
      if not desc_id in dur_dict:
        continue
      
      label_dict[doc_id] = dict()
      dur = dur_dict[desc_id]['duration_second']
      nframes = frame_dict[doc_id] 
      for ann in anno_dict[desc_id+'.mp4']:
        action_class = ann['Event_Type'] 
        start_sec, end_sec = ann['Temporal_Boundary']
        start, end = int(start_sec / dur * nframes), int(end_sec / dur * nframes)
        label_dict[doc_id][(start, end)] = action_class
    return label_dict
  
  def create_data_list(self, label_dict):
    data_list = []
    for doc_id in sorted(label_dict):
      for span in sorted(label_dict[doc_id]):
        data_list.append([doc_id, span, label_dict[doc_id][span]]) 
    return data_list

  def __getitem__(self, idx):  
    doc_id, span, label = self.data_list[idx]
    action_embedding = self.action_embeddings[self.doc_to_feat[doc_id]]
    action_embedding = torch.FloatTensor(action_embedding[span[0]:span[1]+1])
    
    action_mask = torch.zeros(self.max_frame_num)
    action_mask[:action_embedding.size(0)] = 1.
    action_embedding = fix_embedding_length(action_embedding, self.max_frame_num)
    
    return action_embedding, action_mask, label

  def __len__(self):
    return len(self.data_list)
