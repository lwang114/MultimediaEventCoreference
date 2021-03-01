# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import pyhocon
import os
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
from image_models import ResNet34, ResNet50, ResNet101
from tqdm import tqdm

def get_mention_doc(caption_file, bbox_file, out_prefix):
    '''
    :param caption_file: str of filename of a .txt file. Each line is of the format [doc id] [caption text]
    :param bbox_file: str of filename of a .txt file. Each line is of the format [doc id] [phrase] [x_min, y_min, x_max, y_max]
    :param out_prefix: str of the prefix of the three output files
       1. [out_prefix].json: storing dict of {[doc_id]: list of [sent id, token id, token, is entity/event]}
       2. [out_prefix]_entities.json: storing a list of dicts of 
          {'doc_id': str, doc id,
           'subtopic': '0',
           'm_id': '0',
           'sentence_id': 0,
           'tokens_ids': list of ints, position of mention tokens in the current sentence,
           'tokens': str, mention tokens concatenated with space,
           'tags': '',
           'lemmas': '',
           'cluster_id': cluster id, equal to tokens in this case,
           'cluster_desc': '',
           'singleton': boolean}
       3. [out_prefix]_bboxes.json: storing a list of dicts of
          {'doc_id': str, doc id,
           'subtopic': '0',
           'm_id': '0',
           'sentence_id': 0,
           'bbox': list of ints, [x_min, y_min, x_max, y_max] of the bounding box,
           'tokens': str, tokens concatenated with space,
           'cluster_id': cluster id, equal to [doc_id]_[tokens] in this case,
           } 
    '''
    sen_dicts = {}
    with open(caption_file, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.split()
            doc_id = '{}_{}'.format(parts[0], idx)
            # print('doc id {}'.format(doc_id))
            caption = parts[1:]                

            sen_dicts[doc_id] = []
            for token_id, token in enumerate(caption):
                sen_dicts[doc_id].append([0, token_id, token, True])
    json.dump(sen_dicts, open('{}.json'.format(out_prefix), 'w'), indent=4)

    # Match bbox with caption
    mention_dicts = []
    bbox_dicts = []
    with open(bbox_file, 'r') as bbox_f:
      idx = -1
      prev_doc_id = ''
      for line in tqdm(bbox_f):
        parts = line.split()
        doc_id = parts[0]
        phrase = parts[1:-4] 
        bbox = parts[-4:]
        bbox[0] = int(bbox[0])
        bbox[1] = int(bbox[1])
        bbox[2] = int(bbox[2])
        bbox[3] = int(bbox[3]) 

        if doc_id != prev_doc_id:
          idx += 1
          prev_doc_id = doc_id
          # if idx > 10: # XXX
          #   break
        bbox_dicts.append({'doc_id': '{}_{}'.format(doc_id, idx),
                           'subtopic': '0',
                           'm_id': '0',
                           'sentence_id': 0,
                           'bbox': bbox,
                           'tokens': ' '.join(phrase),
                           'cluster_id': '{}_{}'.format(doc_id, '_'.join(phrase))
                           })

        tokens = sen_dicts['{}_{}'.format(doc_id, idx)]
        phrase_start, phrase_end = -1, -1
        phrase_len = len(phrase)
        sent_len = len(tokens)
        for token_id in range(sent_len-phrase_len+1):
          cur_mention = ' '.join([token[2] for token in tokens[token_id:token_id+phrase_len]])
          if ' '.join(phrase) == cur_mention:
            phrase_start = token_id
            phrase_end = token_id+phrase_len
            mention_dicts.append({'doc_id': '{}_{}'.format(doc_id, idx),
                                  'subtopic': '0',
                                  'm_id': '0',
                                  'sentence_id': 0,
                                  'tokens_ids': list(range(phrase_start, phrase_end)),
                                  'tokens': ' '.join(phrase),
                                  'cluster_id': '{}_{}'.format(doc_id, '_'.join(phrase))
                                  })
            break
        if phrase_start == -1:
          print('bbox not found', doc_id, phrase)
    json.dump(bbox_dicts, open('{}_bboxes.json'.format(out_prefix), 'w'), indent=4)
    json.dump(mention_dicts, open('{}_entities.json'.format(out_prefix), 'w'), indent=4)

def get_mention_doc_concat(caption_file, phrase_file, bbox_file, test_id_file, data_dir, split='train'):
    '''
    :param caption_file: str of filename of a .txt file. Each line is of the format [doc id] [caption text]
    :param bbox_file: str of filename of a .txt file. Each line is of the format [doc id] [phrase] [x_min, y_min, x_max, y_max]
    :param out_prefix: str of the prefix of the three output files
       1. [out_prefix].json: storing dict of {[doc_id]: list of [sent id, token id, token, is entity/event]}
       2. [out_prefix]_entities.json: storing a list of dicts of 
          {'doc_id': str, doc id,
           'subtopic': '0',
           'm_id': '0',
           'sentence_id': 0,
           'tokens_ids': list of ints, position of mention tokens in the current sentence,
           'tokens': str, mention tokens concatenated with space,
           'tags': '',
           'lemmas': '',
           'cluster_id': cluster id, equal to tokens in this case,
           'cluster_desc': '',
           'singleton': boolean}
       3. [out_prefix]_bboxes.json: storing a list of dicts of
          {'doc_id': str, doc id,
           'subtopic': '0',
           'm_id': '0',
           'sentence_id': 0,
           'bbox': list of ints, [x_min, y_min, x_max, y_max] of the bounding box,
           'tokens': str, tokens concatenated with space,
           'cluster_id': cluster id, equal to [doc_id]_[tokens] in this case,
           } 
    '''
    test_ids = [] 
    with codecs.open(test_id_file, 'r') as f:
      for line in f:
        test_ids.append('_'.join(line.split('_')[:-1]))

    sen_dicts = {}
    sen_start_dicts = {}
    with open(caption_file, 'r') as f:
        prev_id = ''
        idx = -1
        sen_start = 0
        for line in f:
            parts = line.split()
            doc_id = parts[0].split('.')[0]
            if (split == 'train' and doc_id in test_ids) or (split == 'test' and not doc_id in test_ids):
              continue 
            sent_id = int(parts[1])
            # print('doc id {}'.format(doc_id))
            caption = parts[2:]                
            if doc_id != prev_id:
              prev_id = doc_id
              sen_start = 0
              sen_dicts[doc_id] = []
              sen_start_dicts[doc_id] = []
              idx += 1
              print(idx, doc_id)

            sen_start_dicts[doc_id].append(sen_start)
            for token_id, token in enumerate(caption):
                sen_dicts[doc_id].append([sent_id, sen_start+token_id, token, True])
            sen_start += len(caption)
    json.dump(sen_dicts, open('{}/{}.json'.format(data_dir, split), 'w'), indent=4)

    # Extract and save bbox info
    mention_dicts = []
    bbox_dicts = []
    cluster2phrase = {}
    with open(phrase_file, 'r') as phrase_f:
      idx = -1
      prev_id = ''
      for line in phrase_f:
        parts = line.split()
        doc_id = parts[0].split('.')[0]
        if (split == 'train' and doc_id in test_ids) or (split == 'test' and not doc_id in test_ids):
            continue
        sent_id = int(parts[1])
        cluster_id = parts[2]
        if cluster_id == '0':
          continue
        if doc_id != prev_id:
          idx += 1
          prev_id = doc_id
          # print(idx, doc_id, cluster_id) # XXX
        phrase = parts[3:-1]
        phrase_start = sen_start_dicts[doc_id][sent_id-1] + int(parts[-1]) - 1
        phrase_end = phrase_start + len(phrase)
        mention_dicts.append({'doc_id': doc_id,
                              'subtopic': '0',
                              'm_id': '0',
                              'sentence_id': sent_id,
                              'tokens_ids': list(range(phrase_start, phrase_end)),
                              'tokens': ' '.join(phrase),
                              'cluster_id': cluster_id
                              })
        cluster2phrase[cluster_id] = ' '.join(phrase)

    with open(bbox_file, 'r') as bbox_f:
      idx = -1
      prev_id = ''
      for line in bbox_f:
        parts = line.split()
        doc_id = parts[0].split('.')[0]
        if (split == 'train' and doc_id in test_ids) or (split == 'test' and not doc_id in test_ids):
            continue
        cluster_id = parts[1]
        if doc_id != prev_id:
          idx += 1
          prev_id = doc_id
        bbox = parts[-4:]
        bbox_dicts.append({'doc_id': doc_id,
                           'bbox': [int(x) for x in bbox],
                           'tokens': cluster2phrase.get(cluster_id, ''),
                           'cluster_id': cluster_id
                           })
    bbox_dicts = sorted(bbox_dicts, key=lambda x:x['doc_id'])
    mention_dicts = sorted(mention_dicts, key=lambda x:x['doc_id'])
    json.dump(bbox_dicts, open('{}/{}_bboxes.json'.format(data_dir, split), 'w'), indent=4)
    json.dump(mention_dicts, open('{}/{}_entities.json'.format(data_dir, split), 'w'), indent=4)
    json.dump(mention_dicts, open('{}/{}_mixed.json'.format(data_dir, split), 'w'), indent=4)


def match_rcnn_embeddings(bbox_gold_file, bbox_rcnn_file, rcnn_feat_file, split='train'):
  def _IoU(b1, b2):
    iou = 0.
    x_minmin = min(b1[0], b2[0])
    x_minmax = max(b1[0], b2[0])
    y_minmin = min(b1[1], b2[1])
    y_minmax = max(b1[1], b2[1])
    x_maxmin = min(b1[2], b2[2])
    x_maxmax = max(b1[2], b2[2])
    y_maxmin = max(b1[3], b2[3])
    y_maxmax = max(b1[3], b2[3])

    if y_maxmin < y_minmax or x_maxmin < x_minmax:
      return 0
    else:
      S_i = (y_maxmin - y_minmax) * (x_maxmin - x_minmax)
      S_o = (y_maxmax - y_minmin) * (x_maxmax - x_minmin) 
      return S_i / S_o

  # gold_box_json = '{}/{}_bboxes.json'.format(config['data_folder'], split)
  gold_box_dicts = json.load(codecs.open(bbox_gold_file, 'r', 'utf-8'))
  rcnn_feats = np.load(rcnn_feat_file)

  pred_box_dicts = []
  matched_feats = {}
  prev_id = ''
  for line in open(bbox_rcnn_file, 'r'):
    parts = line.split()
    doc_id, capt_id = parts[0].split('.jpg_')
    if capt_id != '1':
      continue

    if prev_id != doc_id:
      matched_feats[doc_id] = [] 
      prev_id = doc_id
    rcnn_feat = rcnn_feats[parts[0]]
    pred_box = [int(x) for x in parts[-4:]]
    
    match_feat = []
    for gold_box_dict in gold_box_dicts[:10]: # XXX
      gold_box = gold_box_dict['bbox']
      if gold_box_dict['doc_id'] == doc_id and _IoU(pred_box, gold_box) > 0.5:
        pred_box_dicts.append({'doc_id': doc_id,
                         'bbox': pred_box,
                         'tokens': gold_box['tokens'],
                         'cluster_id': gold_box['cluster_id']})
        matched_feats[doc_id].append(rcnn_feat)
  
  matched_feats = {k: np.stack(feat) for k, feat in matched_feats.items()}  
  np.savez('{}_matched_rcnn.npz'.format(split), **matched_feats)

def extract_image_embeddings(config, split='train'):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  bbox_json = '{}/{}_bboxes.json'.format(config['data_folder'], split)
  bbox_dicts = json.load(codecs.open(bbox_json, 'r', 'utf-8'))
  cluster_dict = {}
  feat_dict = {}
  transform = transforms.Compose([
              transforms.Resize(256),
              # transforms.RandomHorizontalFlip(),
              # transforms.RandomCrop(224),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))])

  if config['img_feat_type'] == 'resnet101':
    image_model = ResNet101(device=device) 
  elif config['img_feat_type'] == 'resnet50':
    image_model = ResNet50(device=device) 
  elif config['img_feat_type'] == 'resnet34':
    image_model = ResNet34(device=device) 

  prev_id = ''
  idx = -1
  for bbox_dict in bbox_dicts:
    doc_id = bbox_dict['doc_id']
    if prev_id != doc_id:
      prev_id = doc_id
      idx += 1
      if idx > 30: # XXX
        break

    cluster_id = bbox_dict['cluster_id']
    bbox = bbox_dict['bbox']
    # Load image
    img = Image.open(os.path.join(config['image_dir'], doc_id+'.jpg'))

    # Load bbox
    if not cluster_id in cluster_dict:
      cluster_dict[cluster_id] = len(cluster_dict)
    else:
      continue
    region = img.crop(box=bbox)
    region = transform(region)
    region = region.to(device)

    # Extract feature
    if config['img_feat_type'] == 'resnet101':
      _, feat, _ = image_model(region.unsqueeze(0), return_feat=True) 
    else:
      _, feat = image_model(region.unsqueeze(0), return_feat=True)
    feat_id = '{}_{}'.format(doc_id, idx)
    if not feat_id in feat_dict:
      feat_dict[feat_id] = [feat.cpu().detach().numpy()]
    else:
      feat_dict[feat_id].append(feat.cpu().detach().numpy())
  
  feat_dict = {k:np.concatenate(feat_dict[k]) for k in feat_dict}
  np.savez('{}/{}_{}.npz'.format(config['data_folder'], split, config['img_feat_type']), **feat_dict)


def extract_glove_embeddings(config, glove_file, split='test'):
    ''' Extract glove embeddings for a sentence
    :param doc_json: json metainfo file in m2e2 format
    :return out_prefix: output embedding for the sentences
    '''
    doc_json = os.path.join(config['data_folder'], '{}.json'.format(split))
    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    print('Number of documents: {}'.format(len(documents)))
    vocab = {'$$$UNK$$$': 0}
    # Compute vocab of the documents
    for doc_id in sorted(documents): # XXX
        tokens = documents[doc_id]
        for token in tokens:
            if not token[2].lower() in vocab:
                # print(token[2].lower())
                vocab[token[2].lower()] = len(vocab)
    print('Vocabulary size: {}'.format(len(vocab)))
                
    embed_matrix = [[0.0] * 300] 
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
                # print('Found {}'.format(word))
                embed= [float(x) for x in segments[-300:]]
                embed_matrix.append(embed)
                vocab_emb[word] = len(vocab_emb)
    print('Vocabulary size with embeddings: {}'.format(len(vocab_emb)))
    json.dump(vocab_emb, 
              codecs.open(os.path.join(config['data_folder'], split+'_vocab.json'), 'w', 'utf-8'), 
              indent=4)
    
    # Convert the documents into embedding sequence
    doc_embeddings = {}
    for idx, doc_id in enumerate(sorted(documents)): # XXX
        embed_id = '{}_{}'.format(doc_id, idx)
        print(embed_id)
        tokens = documents[doc_id]
        doc_embedding = []
        for token in tokens:
            token_id = vocab_emb.get(token[2].lower(), 0)
            doc_embedding.append(embed_matrix[token_id])
        print(doc_id, len(tokens), np.asarray(doc_embedding).shape)
        doc_embeddings[embed_id] = np.asarray(doc_embedding)
    np.savez(os.path.join(config['data_folder'], split+'_glove_embeddings.npz'), **doc_embeddings)

def convert_image_embedding(in_file, img_ids, out_file):
  in_feats = np.load(in_file)
  imgid2idx = {}
  out_feats = {}
  feat_keys = sorted(in_feats, key=lambda x:int(x.split('_')[-1]))
  for idx, img_id in enumerate(img_ids):
    imgid2idx[img_id] = len(imgid2idx)
    for feat_key in feat_keys:
      cur_img_id = feat_key.split('.jpg')[0]
      if img_id == cur_img_id:
        print('{}_{}'.format(img_id, imgid2idx[img_id]))
        out_feats['{}_{}'.format(img_id, imgid2idx[img_id])] = in_feats[feat_key]
        break
  np.savez(out_file, **out_feats)  
    
if __name__ == '__main__':
    config_file = 'configs/config_grounded_supervised_flickr_glove_lstm.json'
    caption_file = '/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/flickr8k_sentences.txt'
    phrase_file = '/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/flickr8k_phrases.txt'
    bbox_file = '/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/flickr8k_bboxes.txt'
    glove_file = 'm2e2/data/glove/glove.840B.300d.txt'
    
    config = pyhocon.ConfigFactory.parse_file(config_file)
    if not os.path.isdir(config['data_folder']):
        os.makedirs(config['data_folder'])

    # get_mention_doc(caption_file, bbox_file, os.path.join(config['data_folder'], 'flickr'))
    # get_mention_doc_concat(caption_file, phrase_file, bbox_file, config['test_id_file'], config['data_folder'], 'train')
    # get_mention_doc_concat(caption_file, phrase_file, bbox_file, config['test_id_file'], config['data_folder'], 'test')
    # extract_glove_embeddings(config, glove_file, split='train')
    # extract_glove_embeddings(config, glove_file, split='test')
    # extract_image_embeddings(config, split='train')
    # extract_image_embeddings(config, split='test')
    # img_ids = sorted(json.load(open(os.path.join(config['data_folder'], 'train.json'), 'r')))
    # convert_image_embedding(os.path.join(config['image_dir'], '../flickr30k_rcnn.npz'), img_ids,
    #                         os.path.join(config['data_folder'], 'train_rcnn.npz'))
    # img_ids = sorted(json.load(open(os.path.join(config['data_folder'], 'test.json'), 'r')))
    # convert_image_embedding(os.path.join(config['image_dir'], '../flickr30k_rcnn.npz'), img_ids,
    #                         os.path.join(config['data_folder'], 'test_rcnn.npz'))
    bbox_gold_file = 'data/flickr30k/mentions/flickr_bboxes.json' 
    bbox_rcnn_file = '../../data/flickr30k/flickr30k_bboxes_rcnn.txt' 
    rcnn_feat_file = '../../data/flickr30k/flickr30k_rcnn.npz'
    match_rcnn_embeddings(bbox_gold_file, bbox_rcnn_file, rcnn_feat_file)
