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
from transformers import AdamW, get_linear_schedule_with_warmup
from models import SpanEmbedder, SimplePairWiseClassifier
from grounded_coreference import GroundedCoreferencer, ResNet152 
from corpus import GroundingDataset
from evaluator import Evaluation


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
        return optim.SGD(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)

def get_pairwise_labels(labels, is_training, device):
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
    text_model.load_state_dict(torch.load('{}/models/text_model.{}.pth'.format(args.exp_dir, args.start_epoch)))
    image_model.load_state_dict(torch.load('{}/models/image_model.{}.pth'.format(args.exp_dir, args.start_epoch)))
    coref_model.load_state_dict(torch.load('{}/models/text_scorer.{}.pth'.format(args.exp_dir, args.start_epoch)))

  # Set up the optimizer  
  optimizer = get_optimizer(config, [text_model, image_model, coref_model])
   
  # Start training
  if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']: 
    text_model.train()

  image_model.train()
  coref_model.train()
  total_loss = 0.
  total = 0.
  begin_time = time.time()
  for epoch in range(args.start_epoch, config.epochs):
    for i, batch in enumerate(train_loader):
      start_end_embeddings, continuous_embeddings,\
      span_mask, width, videos, video_mask, _ = batch   

      B = start_end_embeddings.size(0)
      start_end_embeddings = start_end_embeddings.to(device)
      continuous_embeddings = continuous_embeddings.to(device)
      span_mask = span_mask.to(device)
      width = width.to(device)
      videos = videos.to(device)
      video_mask = video_mask.to(device)
      optimizer.zero_grad()

      text_output = text_model(start_end_embeddings, continuous_embeddings, width)
      L = videos.size(1)
      W, H, C = videos.size(2), videos.size(3), videos.size(4)
      video_output = image_model(videos.view(B*L, W, H, C))\
                     .view(B, L*video_output.size(2)*video_output.size(3), video_output.size(4)) 
      video_mask = video_mask.unsqueeze(-1).repeat(1, 1, video_output.size(2)*video_output.size(3)).view(B, -1)
      loss = coref_model(text_output, video_output, span_mask, video_mask)
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

    torch.save(text_model.state_dict(), '{}/models/text_model.{}.pth'.format(args.exp_dir, args.start_epoch))
    torch.save(image_model.state_dict(), '{}/models/image_model.{}.pth'.format(args.exp_dir, args.start_epoch))
    torch.save(coref_model.text_scorer.state_dict(), '{}/models/text_scorer.{}.pth'.format(args.exp_dir, args.start_epoch))
 
    if epoch % 5 == 0:
      all_scores = []
      all_labels = []
      with torch.no_grad(): 
        for i, batch in enumerate(test_loader):
          start_end_embeddings, continuous_embeddings,\
          span_mask, width, videos, video_mask, labels = batch   
          start_end_embeddings = start_end_embeddings.to(device)
          continuous_embeddings = continuous_embeddings.to(device)
          span_mask = span_mask.to(device)
          span_num = span_mask.sum(1)
          width = width.to(device)
          videos = videos.to(device)
          video_mask = video_mask.to(device)
          labels = labels.to(device)

          # Extract span and video embeddings
          text_output = text_model(start_end_embeddings, continuous_embeddings, width)
          L = videos.size(1)
          W, H, C = videos.size(2), videos.size(3), videos.size(4)
          video_output = image_model(videos.view(B*L, W, H, C))\
          .view(B, L*video_output.size(2)*video_output.size(3), video_output.size(4)) 
          video_mask = video_mask.unsqueeze(-1).repeat(1, 1, video_output.size(2)*video_output.size(3)).view(B, -1)
         
          # Compute score for each span pair
          B = start_end_embeddings.size(0) 
          for idx in range(B):
            first_idx, second_idx, pairwise_labels = get_pairwise_labels(labels[idx, :span_num[idx]], is_training=False, device=device)
            scores = coref_model.predict(text_output[idx, first_idx], video_output[idx, first_idx],\
                                       text_mask[idx, first_idx], video_mask[idx, first_idx],\
                                       text_output[idx, second_idx], video_output[idx, second_idx],\
                                       span_mask[idx, first_idx], video_mask[idx, second_idx], batched=False)
            all_scores.append(scores.squeeze(0))
            all_labels.append(pairwise_labels.to(torch.int)) 

        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)

        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))
        print('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        print('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                       len(all_labels)))
        print('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                  eval.get_precision(), eval.get_f1()))
        logger.info('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        logger.info('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                             len(all_labels)))
        logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                        eval.get_precision(), eval.get_f1()))
        f1.append(eval.get_f1())


 
if __name__ == '__main__':
  # Set up argument parser
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', type=str, default='models/grounded_coref')
  parser.add_argument('--config', type=str, default='configs/config_grounded.json')
  parser.add_argument('--start_epoch', type=int, default=0)
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if not os.path.isdir(args.exp_dir):
    os.mkdir(args.exp_dir)

  # Set up logger
  config = pyhocon.ConfigFactory.parse_file(args.config)
  if not os.path.isdir(config['log_path']):
    os.mkdir(config['log_path']) 
  logging.basicConfig(filename=os.path.join(config['log_path'],'{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                      format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) 

  # Initialize dataloaders
  train_set = GroundingDataset(os.path.join(config['data_folder'], 'train.json'), os.path.join(config['data_folder'], 'train_mixed.json'), config)
  test_set = GroundingDataset(os.path.join(config['data_folder'], 'train.json'), os.path.join(config['data_folder'], 'train_mixed.json'), config) # XXX
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

  # Initialize models
  text_model = SpanEmbedder(config, device).to(device)
  image_model = ResNet152(embedding_dim=config['hidden_layer'])
  coref_model = GroundedCoreferencer(config).to(device)

  # Training
  train(text_model, image_model, coref_model, train_loader, test_loader, args)
