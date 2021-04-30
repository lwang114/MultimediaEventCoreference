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
from text_models import NoOp, BiLSTM
from corpus_crossmedia import VideoM2E2SupervisedCrossmediaDataset
from evaluator import Evaluation

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


def train(text_model, visual_model, train_loader, test_loader, args):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(True)
  
  if not isinstance(text_model, torch.nn.DataParallel):
    text_model = nn.DataParallel(text_model)

  if not isinstance(visual_model, torch.nn.DataParallel):
    visual_model = nn.DataParallel(visual_model)

  text_model.to(device)
  visual_model.to(device)
  
  # Create/load exp
  if not os.path.isdir(args.exp_dir):
    os.path.mkdir(args.exp_dir)

  # Define the training criterion
  criterion = nn.BCEWithLogitsLoss()

  # Set up the optimizer  
  optimizer = get_optimizer(config, [text_model, visual_model])

  # Start training
  total_loss = 0.
  total = 0.
  begin_time = time.time()
  if args.evaluate_only:
    config.epochs = 0

  for epoch in range(config.epochs):
    text_model.train()
    visual_model.train()
    
    for i, batch in enumerate(train_loader):
      mention_embeddings,\
      action_embeddings,\
      action_masks,\
      coref_labels = batch 

      B = mention_embeddings.size(0)
      mention_embeddings = mention_embeddings.to(device)
      action_embeddings = action_embeddings.to(device)
      action_masks = action_masks.to(device)
      coref_labels = coref_labels.to(device)

      optimizer.zero_grad()

      mention_output = text_model(mention_embeddings)
      action_output = visual_model(action_embeddings).sum(dim=1) / action_masks.sum(-1).unsqueeze(-1)
      scores = (mention_output * action_output).sum(dim=-1)
      loss = criterion(scores, coref_labels.float())
      loss.backward()
      optimizer.step()

      total_loss += loss.item() * B
      total += B
      if i % 100 == 0:
        info = 'Iter {} {:.4f}'.format(i, total_loss / total)
        print(info)
        logging.info(info) 
    
    info = 'Epoch: [{}][{}/{}]\tTime {:.3f}\tLoss total {:.4f} ({:.4f})'.format(epoch, i, len(train_loader), time.time()-begin_time, total_loss, total_loss / total)
    print(info)
    logging.info(info)

    torch.save(text_model.module.state_dict(), '{}/text_model.{}.pth'.format(args.exp_dir, epoch))
    torch.save(visual_model.module.state_dict(), '{}/visual_model.{}.pth'.format(args.exp_dir, epoch))
 
    if epoch % 5 == 0:
      test(text_model, visual_model, test_loader, args)

  if args.evaluate_only:
    test(text_model, visual_model, test_loader, args)


def test(text_model, visual_model, test_loader, args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  config = pyhocon.ConfigFactory.parse_file(args.config)
  all_scores = []
  all_labels = []
  
  text_model.eval()
  visual_model.eval()

  with torch.no_grad():
    pred_dicts = []
    f_out = open(os.path.join(config['model_path'], 'prediction.readable'), 'w') 
    f_out.write('Document id\tmention info\taction info\tscore\tlabel\n')
    for i, batch in enumerate(test_loader):
      mention_embedding,\
      action_embedding,\
      action_mask,\
      coref_label = batch
      
      mention_embedding = mention_embedding.to(device)
      action_embedding = action_embedding.to(device)
      action_mask = action_mask.to(device)
      coref_label = coref_label.to(device)

      mention_output = text_model(mention_embedding)
      action_output = visual_model(action_embedding).sum(dim=1) / action_mask.sum(-1).unsqueeze(-1)
      score = (mention_output * action_output).sum(axis=-1)
      all_scores.append(score)
      all_labels.append(coref_label)
      for idx in range(mention_embedding.size(0)):
        global_idx = i * test_loader.batch_size + idx
        _, doc_id, m_info, a_info = test_loader.dataset.data_list[global_idx]    
        f_out.write(f'{doc_id}\t{m_info}\t{a_info}\t{score}\t{coref_label}\n')

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    preds = (all_scores > 0).to(torch.int)
    eval = Evaluation(preds, all_labels)
    print('Number of predictions: {}/{}'.format(preds.sum(), len(preds)))
    print('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()), 
                                                   len(all_labels)))
    print('Pairwise - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                eval.get_precision(),
                                                                eval.get_f1()))

if __name__ == '__main__':
  # Set up argument parser
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', type=str, default='')
  parser.add_argument('--config', type=str, default='configs/config_crossmedia_coref.json')
  parser.add_argument('--evaluate_only', action='store_true')
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
  train_set = VideoM2E2SupervisedCrossmediaDataset(config, split='train')
  test_set = VideoM2E2SupervisedCrossmediaDataset(config, split='test')

  fix_seed(config)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

  # Initialize models
  text_model = nn.Sequential(
                  nn.Linear(1024, 800),
                  nn.ReLU(),
                  nn.Linear(800, 800),
                  nn.ReLU(),
                  nn.Linear(800, 800),
               )
  visual_model = BiLSTM(400, 400, num_layers=3)

  # Training
  train(text_model, visual_model, train_loader, test_loader, args)
