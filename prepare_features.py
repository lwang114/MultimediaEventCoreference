import torch
import numpy as np
import argparse
import pyhocon
import os
import logging
import cv2
import codecs
import json
import math
# XXX
# import torchvision.transforms as transforms
import PIL.Image as Image
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from image_models import ResNet101
from coref.model_utils import pad_and_read_bert
from coref.utils import create_corpus

logger = logging.getLogger(__name__)
def load_video(filename, config, transform=None, image_prefix=None):
    '''Load video
    :param filename: str, video filename
    :return video_frames: FloatTensor of size (batch size, max num. of frames, width, height, n_channel)
    :return mask: LongTensor of size (batch size, max num. of frames)
    '''    
    max_frame_num = config.max_frame_num
    # Create mask
    mask = torch.ones((max_frame_num,))

    # Load video
    try:
      cap = cv2.VideoCapture(filename)
      frame_rate = cap.get(5) 
      video = [] 
      while True:
        frame_id = cap.get(1)
        ret, img = cap.read()
        if not ret:
          print('{}, frame_rate, number of video frames: {}, {}'.format(filename, frame_rate, len(video)))
          break
        if (frame_id % math.floor(frame_rate) == 0):
          video.append(img)    

      # Subsample the video frames
      step = len(video) // max_frame_num
      indices = list(range(0, step*max_frame_num, step))
    except:
      print('Corrupted video file: {}'.format(filename))
      logging.info('Corrupted video file: {}'.format(filename))
      video = [torch.zeros((1, 3, 224, 224)) for _ in range(max_frame_num)]
      return torch.cat(video, dim=0), mask

    if not image_prefix is None:
      for idx, img in enumerate(video):
          img = Image.fromarray(img)
          img.save('{}_{:03d}.jpg'.format(image_prefix, idx))

    video = [Image.fromarray(video[idx]) for idx in indices]
    # Apply transform to each frame
    if transform is not None:
      video = [transform(img).unsqueeze(0) for img in video]
    
    return torch.cat(video, dim=0), mask

def save_frame_rate(config, split='train'):
  doc_json = os.path.join(config['data_folder'], split+'.json')
  documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
  doc_ids = sorted(documents)
  out_fn = os.path.join(config['data_folder'], '{}_framerates.txt'.format(split))
  out_f = open(out_fn, 'w') 

  for idx, doc_id in enumerate(doc_ids):
    video_file = os.path.join(config['image_dir'], doc_id+'.mp4')
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(5) 
    print(doc_id, frame_rate) # XXX
    out_f.write('{} {}\n'.format(doc_id, frame_rate))
  out_f.close()

def extract_glove_embeddings(config, split, glove_file, dimension=300, out_prefix='glove_embedding'):
    ''' Extract glove embeddings for a sentence
    :param doc_json: json metainfo file in m2e2 format
    :return out_prefix: output embedding for the sentences
    '''
    doc_json = os.path.join(config['data_folder'], split+'.json')
    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    if config.test_id_file and split == 'test':
        with open(config.test_id_file, 'r') as f:
            test_ids = sorted(f.read().strip().split('\n'), key=lambda x:int(x.split('_')[-1].split('.')[0]))
            test_ids = ['_'.join(k.split('_')[:-1]) for k in test_ids]
        documents = {test_id:documents[test_id] for test_id in test_ids}
    else:
        documents = cleanup(documents, config)
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
                # print('Found {}'.format(word))
                embed= [float(x) for x in segments[-300:]]
                embed_matrix.append(embed)
                vocab_emb[word] = len(vocab_emb)
    print('Vocabulary size with embeddings: {}'.format(len(vocab_emb)))
    json.dump(vocab_emb, open(out_prefix+'_vocab.json', 'w'), indent=4, sort_keys=True)
    
    # Convert the documents into embedding sequence
    doc_embeddings = {}
    for idx, doc_id in enumerate(sorted(documents)): # XXX
        embed_id = '{}_{}'.format(doc_id, idx)
        # print(embed_id)
        tokens = documents[doc_id]
        doc_embedding = []
        for token in tokens:
            token_id = vocab_emb.get(token[2].lower(), 0)
            doc_embedding.append(embed_matrix[token_id])
        print(np.asarray(doc_embedding).shape)
        doc_embeddings[embed_id] = np.asarray(doc_embedding)
    np.savez(out_prefix+'.npz', **doc_embeddings)

def extract_bert_embeddings(config, split, out_prefix='bert_embedding'):
    device = torch.device('cuda:{}'.format(config.gpu_num[0]))
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    data = create_corpus(config, bert_tokenizer, split)
    doc_json = os.path.join(config['data_folder'], split+'.json')
    filtered_doc_ids = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    # XXX cleanup(json.load(codecs.open(doc_json, 'r', 'utf-8')), config)
    
    list_of_doc_id_tokens = []    
    for topic_num in tqdm(range(len(data.topic_list))):  
        list_of_doc_id_tokens_topic = [(doc_id, bert_tokens)
                                       for doc_id, bert_tokens in
                                       zip(data.topics_list_of_docs[topic_num], data.topics_bert_tokens[topic_num])]
        list_of_doc_id_tokens.extend(list_of_doc_id_tokens_topic)
    
    if 'test_id_file' in config and split == 'test':
        with open(config.test_id_file, 'r') as f:
            test_ids = f.read().strip().split('\n')
            test_ids = ['_'.join(k.split('_')[:-1]) for k in test_ids]
        list_of_doc_id_tokens = [doc_id_token for doc_id_token in list_of_doc_id_tokens if doc_id_token[0] in test_ids]
            
    list_of_doc_id_tokens = sorted(list_of_doc_id_tokens, key=lambda x:x[0]) # XXX
    print('Number of documents: {}'.format(len(list_of_doc_id_tokens)))

    emb_ids = {}
    docs_embeddings = {}
    with torch.no_grad():
        total = len(list_of_doc_id_tokens)
        for i in range(total):
            doc_id = list_of_doc_id_tokens[i][0]
            if not doc_id in emb_ids and doc_id in filtered_doc_ids:
                emb_ids[doc_id] = '{}_{}'.format(doc_id, len(emb_ids))
        
        nbatches = total // config['batch_size'] + 1 if total % config['batch_size'] != 0 else total // config['batch_size']
        for b in range(nbatches):
            start_idx = b * config['batch_size']
            end_idx = min((b + 1) * config['batch_size'], total)
            batch_idxs = list(range(start_idx, end_idx))
            doc_ids = [list_of_doc_id_tokens[i][0] for i in batch_idxs]
            bert_tokens = [list_of_doc_id_tokens[i][1] for i in batch_idxs]
            bert_embeddings, docs_length = pad_and_read_bert(bert_tokens, bert_model)
            for idx, doc_id in enumerate(doc_ids):
                if not doc_id in emb_ids:
                    print('Skip {}'.format(doc_id))
                    continue
                emb_id = emb_ids[doc_id]
                bert_embedding = bert_embeddings[idx][:docs_length[idx]].cpu().detach().numpy()
                if emb_id in docs_embeddings:
                    print(doc_id, emb_id) # XXX
                    docs_embeddings[emb_id] = np.concatenate([docs_embeddings[emb_id], bert_embedding], axis=0)
                else:
                    docs_embeddings[emb_id] = bert_embedding
    np.savez(out_prefix+'.npz', **docs_embeddings)
    
def cleanup(documents, config):
    filtered_documents = {}
    img_ids = [img_id.split('.')[0] for img_id in os.listdir(config['image_dir'])]
    # config['image_dir'] = os.path.join(config['data_folder'], 'train_resnet152') # XXX
    img_files = os.listdir(config['image_dir'])

    if img_files[0].split('.')[-1] == '.jpg':
      img_ids = [img_id.split('.jpg')[0] for img_id in img_files]
    elif img_files[0].split('.')[-1] == '.npy':
      img_ids = ['_'.join(img_id.split('_')[:-1]) for img_id in os.listdir(config['image_dir'])]     

    for doc_id in sorted(documents): 
        filename = os.path.join(config['image_dir'], doc_id+'.mp4')
        if os.path.exists(filename):
            filtered_documents[doc_id] = documents[doc_id]
        elif doc_id in img_ids:
            filtered_documents[doc_id] = documents[doc_id]
    print('Keep {} out of {} documents'.format(len(filtered_documents), len(documents)))
    return filtered_documents


def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='configs/config_grounded.json')
  parser.add_argument('--split', choices={'train', 'test'}, default='train')
  parser.add_argument('--task', type=int)
  args = parser.parse_args()  
  config = pyhocon.ConfigFactory.parse_file(args.config) 
  if not os.path.isdir(config['log_path']):
    os.mkdir(config['log_path']) 
  tasks = [args.task]

  if 0 in tasks:
    if not os.path.isdir(os.path.join(config['data_folder'], args.split+'_resnet101/')):
      os.mkdir(os.path.join(config['data_folder'], args.split+'_resnet101/'))
    if not os.path.isdir(os.path.join(config['data_folder'], args.split+'_video_1fps/')):
      os.mkdir(os.path.join(config['data_folder'], args.split+'_video_1fps/'))
    logging.basicConfig(filename=os.path.join(config['log_path'],'prep_feat_{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)   
    
    # Extract doc ids
    doc_json = os.path.join(config['data_folder'], args.split+'.json')
    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))   
    documents = cleanup(documents, config)
    doc_ids = sorted(documents) # XXX

    # Initialize image model
    transform = transforms.Compose([
              transforms.Resize(256),
              transforms.RandomHorizontalFlip(),
              transforms.RandomCrop(224),
              transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))])
    image_model = ResNet101(device=torch.device('cuda')) 

    for idx, doc_id in enumerate(doc_ids):
      embed_file = os.path.join(config['data_folder'], '{}_resnet101/{}_{}.npy'.format(args.split, doc_id, idx))
      video_file = os.path.join(config['image_dir'], doc_id+'.mp4')
      image_prefix = os.path.join(config['data_folder'], '{}_video_1fps/{}'.format(args.split, doc_id))
      # if os.path.exists(embed_file) and os.path.exists(image_prefix+'_0.png'): # XXX
      if os.path.exists(embed_file):
        print('Skip {}_{}'.format(doc_id, idx))
        continue
      videos, video_mask = load_video(video_file, config, transform, image_prefix)
      video_output, video_feat, video_mask = image_model(videos.unsqueeze(0), video_mask.unsqueeze(0), return_feat=True)
      print('{}_{}'.format(doc_id, idx))
      np.save(embed_file, video_feat.squeeze(0).cpu().detach().numpy())
  if 1 in tasks:
    glove_file = 'm2e2/data/glove/glove.840B.300d.txt'
    extract_glove_embeddings(config, args.split, glove_file, out_prefix='{}_glove_embeddings'.format(args.split))
  if 2 in tasks:
    extract_bert_embeddings(config, args.split, out_prefix='{}_bert_embeddings'.format(args.split))
  if 3 in tasks:
    save_frame_rate(config)

if __name__ == '__main__':
  main()
