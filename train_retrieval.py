# Some code borrowed from https://github.com/ariecattan/coref/
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import time
from datetime import datetime
import json
import argparse
import pyhocon 
import random
import numpy as np
from itertools import combinations
from transformers import AdamW, get_linear_schedule_with_warmup
from models import SpanEmbedder, SimplePairWiseClassifier
from mml_dotproduct_grounded_coreference import MMLDotProductGroundedCoreferencer, NoOp, BiLSTM
from adaptive_mml_dotproduct_grounded_coreference import AdaptiveMMLDotProductGroundedCoreferencer
from corpus import GroundingFeatureDataset
from corpus_glove import GroundingGloveFeatureDataset
from evaluator import Evaluation, RetrievalEvaluation


logger = logging.getLogger(__name__)
def fix_seed(config):
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)

def get_optimizer(config, models):
    parameters = []
    for model in models:
        parameters += [p for p in model.parameters() if p.requires_grad]

    if config.optimizer == "adam":
        return optim.Adam(parameters, lr=config.learning_rate, weight_decay=config.weight_decay, eps=config.adam_epsilon)
    elif config.optimizer == "adamw":
        return AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay, eps=config.adam_epsilon)
    else:
        return optim.SGD(parameters, momentum=0.9, lr=config.learning_rate, weight_decay=config.weight_decay)

def get_pairwise_labels(labels, is_training, device):
    if len(labels) <= 1:
      return None, None, None
    first, second = zip(*list(combinations(range(len(labels)), 2)))
    first = torch.tensor(first)
    second = torch.tensor(second)
    pairwise_labels = (labels[first] != 0) & (labels[second] != 0) & \
                      (labels[first] == labels[second])

    if is_training:
        positives = (pairwise_labels == 1).nonzero().squeeze()
        positive_ratio = len(positives) / len(first)
        negatives = (pairwise_labels != 1).nonzero().squeeze()
        rands = torch.rand(len(negatives))
        rands = (rands < positive_ratio * 20).to(torch.long)
        sampled_negatives = negatives[rands.nonzero().squeeze()]
        new_first = torch.cat((first[positives], first[sampled_negatives]))
        new_second = torch.cat((second[positives], second[sampled_negatives]))
        new_labels = torch.cat((pairwise_labels[positives], pairwise_labels[sampled_negatives]))
        first, second, pairwise_labels = new_first, new_second, new_labels


    pairwise_labels = pairwise_labels.to(torch.long).to(device)

    if config['loss'] == 'hinge':
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(-1, device=device))
    else:
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(0, device=device))
    torch.cuda.empty_cache()


    return first, second, pairwise_labels    


def train(text_model, image_model, coref_model, train_loader, test_loader, args):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(True)
  fix_seed(config)
  
  if not isinstance(text_model, torch.nn.DataParallel):
    text_model = nn.DataParallel(text_model)

  if not isinstance(image_model, torch.nn.DataParallel):
    image_model = nn.DataParallel(image_model)
  
  if not isinstance(coref_model, torch.nn.DataParallel):
    coref_model = nn.DataParallel(coref_model)

  text_model.to(device)
  image_model.to(device)
  coref_model.to(device)

  # Create/load exp
  if not os.path.isdir(args.exp_dir):
    os.path.mkdir(args.exp_dir)

  if args.start_epoch != 0:
    text_model.load_state_dict(torch.load('{}/text_model.{}.pth'.format(args.exp_dir, args.start_epoch)))
    image_model.load_state_dict(torch.load('{}/image_model.{}.pth'.format(args.exp_dir, args.start_epoch)))
    coref_model.load_state_dict(torch.load('{}/text_scorer.{}.pth'.format(args.exp_dir, args.start_epoch)))

  # Set up the optimizer  
  optimizer = get_optimizer(config, [text_model, image_model, coref_model])
  for p in coref_model.parameters(): # XXX
    print('coref model: ', p.size(), p.device, p.requires_grad) 

  # Start training
  total_loss = 0.
  total = 0.
  begin_time = time.time()
  if args.evaluate_only:
    config.epochs = 0
  for epoch in range(args.start_epoch, config.epochs):
    text_model.train()
    image_model.train()
    coref_model.train()
    for i, batch in enumerate(train_loader):
      doc_embeddings, start_mappings, end_mappings,\
      continuous_mappings, width, video_embeddings,\
      labels, text_mask, span_mask, video_mask = batch 

      B = doc_embeddings.size(0)
      doc_embeddings = doc_embeddings.to(device)
      start_mappings = start_mappings.to(device)
      end_mappings = end_mappings.to(device)
      continuous_mappings = continuous_mappings.to(device)
      width = width.to(device)
      video_embeddings = video_embeddings.to(device)
      labels = labels.to(device)
      text_mask = text_mask.to(device)
      span_mask = span_mask.to(device)
      video_mask = video_mask.to(device)

      optimizer.zero_grad()

      # text_output = text_model(start_end_embeddings, continuous_embeddings, width)
      text_output = text_model(doc_embeddings)
      video_output = image_model(video_embeddings)
      if config.loss == 'adaptive_mml':
        start_embeddings = torch.matmul(start_mappings, doc_embeddings)
        end_embeddings = torch.matmul(end_mappings, doc_embeddings)
        start_end_embeddings = torch.cat([start_embeddings, end_embeddings], dim=-1)
        continuous_embeddings = torch.matmul(continuous_mappings, doc_embeddings.unsqueeze(1))
        loss = coref_model(text_output, video_output, text_mask, video_mask, start_end_embeddings, continuous_embeddings, width, span_mask).mean()
      else:
        loss = coref_model(text_output, video_output, text_mask, video_mask).mean() # topic_labels=topic_labels).mean()
      loss.backward()
      optimizer.step()

      total_loss += loss.item() * B
      total += B
      if i % 50 == 0:
        info = 'Iter {} {:.4f}'.format(i, total_loss / total)
        print(info)
        logger.info(info) 
    
    info = 'Epoch: [{}][{}/{}]\tTime {:.3f}\tLoss total {:.4f} ({:.4f})'.format(epoch, i, len(train_loader), time.time()-begin_time, total_loss, total_loss / total)
    print(info)
    logger.info(info)

    torch.save(text_model.module.state_dict(), '{}/text_model.{}.pth'.format(args.exp_dir, epoch))
    torch.save(image_model.module.state_dict(), '{}/image_model.{}.pth'.format(args.exp_dir, epoch))
    torch.save(coref_model.module.text_scorer.state_dict(), '{}/text_scorer.{}.pth'.format(args.exp_dir, epoch))
 
    if epoch % 5 == 0:
      test_retrieve(text_model, image_model, coref_model, test_loader, args)

  if args.evaluate_only:
    test_retrieve(text_model, image_model, coref_model, test_loader, args)



def test_retrieve(text_model, image_model, coref_model, test_loader, args):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  doc_embeddings = []
  video_embeddings = []
  text_masks = []
  video_masks = []
  text_model.eval()
  image_model.eval()
  coref_model.eval()
  with torch.no_grad():
    pred_dicts = []
    for i, batch in enumerate(test_loader):
      doc_embedding, start_mapping, end_mapping,\
      continuous_mapping, width, video_embedding,\
      label, text_mask, span_mask, video_mask = batch
      text_output = text_model(doc_embedding)
      video_output = image_model(video_embedding)

      text_output = text_output.cpu().detach()
      video_output = video_output.cpu().detach()
      text_mask = text_mask.cpu().detach()
      video_mask = video_mask.cpu().detach()

      doc_embeddings.append(text_output)
      video_embeddings.append(video_output)
      text_masks.append(text_mask)
      video_masks.append(video_mask)
  doc_embeddings = torch.cat(doc_embeddings)
  video_embeddings = torch.cat(video_embeddings)
  text_masks = torch.cat(text_masks)
  video_masks = torch.cat(video_masks) 
  I2S_idxs, S2I_idxs = coref_model.module.retrieve(doc_embeddings, video_embeddings, text_masks, video_masks)
  I2S_eval = RetrievalEvaluation(I2S_idxs)
  S2I_eval = RetrievalEvaluation(S2I_idxs)
  I2S_r1 = I2S_eval.get_recall_at_k(1) 
  I2S_r5 = I2S_eval.get_recall_at_k(5)
  I2S_r10 = I2S_eval.get_recall_at_k(10)
  S2I_r1 = S2I_eval.get_recall_at_k(1) 
  S2I_r5 = S2I_eval.get_recall_at_k(5)
  S2I_r10 = S2I_eval.get_recall_at_k(10)

  print('Number of article-video pairs: {}'.format(I2S_idxs.size(0)))
  print('I2S recall@1={}\tI2S recall@5={}\tI2S recall@10={}'.format(I2S_r1, I2S_r5, I2S_r10)) 
  print('S2I recall@1={}\tS2I recall@5={}\tS2I recall@10={}'.format(S2I_r1, S2I_r5, S2I_r10)) 

if __name__ == '__main__':
  # Set up argument parser
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', type=str, default='')
  parser.add_argument('--config', type=str, default='configs/config_grounded.json')
  parser.add_argument('--start_epoch', type=int, default=0)
  parser.add_argument('--evaluate_only', action='store_true')
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # Set up logger
  config = pyhocon.ConfigFactory.parse_file(args.config)
  if not args.exp_dir:
    args.exp_dir = config['model_path']
  else:
    config['model_path'] = args.exp_dir
    
  if not os.path.isdir(config['model_path']):
    os.mkdir(config['model_path'])
  if not os.path.isdir(config['log_path']):
    os.mkdir(config['log_path']) 
  logging.basicConfig(filename=os.path.join(config['log_path'],'{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                      format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) 

  # Initialize dataloaders
  if not config.get('glove_dimension', None):
    train_set = GroundingFeatureDataset(os.path.join(config['data_folder'], 'train.json'), os.path.join(config['data_folder'], 'train_mixed.json'), config, split='train')
    test_set = GroundingFeatureDataset(os.path.join(config['data_folder'], 'test.json'), os.path.join(config['data_folder'], 'test_mixed.json'), config, split='test')
  else:
    train_set = GroundingGloveFeatureDataset(os.path.join(config['data_folder'], 'train.json'), os.path.join(config['data_folder'], 'train_mixed.json'), config, split='train')
    test_set = GroundingGloveFeatureDataset(os.path.join(config['data_folder'], 'test.json'), os.path.join(config['data_folder'], 'test_mixed.json'), config, split='test')
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

  # Initialize models
  if not config.get('glove_dimension', None):
    input_dim = config.bert_hidden_size
  else:
    input_dim = config['glove_dimension']
  embedding_dim = config.hidden_layer

  text_model = BiLSTM(input_dim, embedding_dim)
  if config['img_feat_type'] == 'mmaction_feat':
    image_model = NoOp() 
  else:
    image_model = BiLSTM(2048, embedding_dim)

  if config['loss'] != 'adaptive_mml':
      coref_model = MMLDotProductGroundedCoreferencer(config).to(device)
      if config['training_method'] in ('pipeline', 'continue'):
        text_model.load_state_dict(torch.load(config['span_repr_path'], map_location=device))
  else:
      coref_model = AdaptiveMMLDotProductGroundedCoreferencer(config).to(device)
      if config['training_method'] in ('pipeline', 'continue'):    
        coref_model.span_repr.load_state_dict(torch.load(config['span_repr_path']))
        coref_model.text_scorer.load_state_dict(torch.load(config['pairwise_scorer_path'], map_location=device))

  # Training
  train(text_model, image_model, coref_model, train_loader, test_loader, args)
