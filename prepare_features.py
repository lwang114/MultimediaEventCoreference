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
import torchvision.transforms as transforms
import PIL.Image as Image
from datetime import datetime
from grounded_coreference import ResNet152

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
    video = [Image.fromarray(video[idx]) for idx in indices]
    if not image_prefix is None:
      for idx, img in enumerate(video):
        img.save('{}_{:03d}.jpg'.format(image_prefix, idx))

    # Apply transform to each frame
    if transform is not None:
      video = [transform(img).unsqueeze(0) for img in video]
    
    return torch.cat(video, dim=0), mask

def extract_glove_embeddings(doc_json, glove_file, config, dimension=300, out_prefix='glove_emb'):
    ''' Extract glove embeddings for a sentence
    :param doc_json: json metainfo file in m2e2 format
    :return out_file: output embedding for the sentences
    '''
    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    documents = cleanup(documents, config)
    vocab = {'$$$UNK$$$': 0}
    # Compute vocab of the documents
    for doc_id in sorted(documents): # XXX
        tokens = documents[doc_id]
        for token in tokens:
            if not token[2].lower() in vocab:
                print(token[2].lower())
                vocab[token[2].lower()] = len(vocab)
    print('Vocabulary size: {}'.format(len(vocab)))
                
    embed_matrix = [[0.0] * dimension] 
    vocab_emb = {'$$$UNK$$$': 0} 
    # Load the embeddings
    with codecs.open(glove_file, 'r', 'utf-8') as f:
        for line in f:
            segments = line.strip().split()
            word = segments[0]
            if word in vocab:
                print('{} found'.format(word))
                embed= [float(x) for x in segments[1:]]
                embed_matrix.append(embed)
                vocab_emb[word] = len(vocab_emb)
    print('Vocabulary size with embeddings: {}'.format(len(vocab_emb)))
    json.dump(vocab_emb, open(out_prefix+'_vocab.json', 'w'), indent=4, sort_keys=True)
    
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
        doc_embeddings[embed_id] = np.asarray(doc_embedding)
    np.savez(out_prefix+'.npz', **doc_embeddings)
        
def cleanup(documents, config):
    filtered_documents = {}
    for doc_id in sorted(documents): 
        filename = os.path.join(config['image_dir'], doc_id+'.mp4')
        if os.path.exists(filename):
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
    if not os.path.isdir(os.path.join(config['data_folder'], args.split+'_resnet152/')):
      os.mkdir(os.path.join(config['data_folder'], args.split+'_resnet152/'))
    if not os.path.isdir(os.path.join(config['data_folder'], args.split+'_videoframes_20/')):
      os.mkdir(os.path.join(config['data_folder'], args.split+'_videoframes_20/'))
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
    image_model = ResNet152(device=torch.device('cuda')) 

    for idx, doc_id in enumerate(doc_ids):
      embed_file = os.path.join(config['data_folder'], '{}_resnet152/{}_{}.npy'.format(args.split, doc_id, idx))
      video_file = os.path.join(config['image_dir'], doc_id+'.mp4')
      image_prefix = os.path.join(config['data_folder'], '{}_videoframes_20/{}'.format(args.split, doc_id))
      if os.path.exists(embed_file) and os.path.exists(image_prefix+'_0.png'): # XXX
        print('Skip {}_{}'.format(doc_id, idx))
        continue
      videos, video_mask = load_video(video_file, config, transform, image_prefix)
      video_output, video_feat, video_mask = image_model(videos.unsqueeze(0), video_mask.unsqueeze(0), return_feat=True)
      print('{}_{}'.format(doc_id, idx))
      np.save(embed_file, video_feat.squeeze(0).cpu().detach().numpy())
  if 1 in tasks:
    doc_json = os.path.join(config['data_folder'], args.split+'.json')
    glove_file = 'm2e2/data/glove/glove.6B.300d.txt'
    extract_glove_embeddings(doc_json, glove_file, config, out_prefix='{}_glove_embeddings'.format(args.split))

if __name__ == '__main__':
  main()
