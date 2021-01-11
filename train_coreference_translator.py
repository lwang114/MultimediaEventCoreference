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
from corpus_text import TextFeatureDataset
from evaluator import Evaluation, RetrievalEvaluation
from conll import write_output_file

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
    B = labels.size(0)
    pairwise_labels_all = []
    first_idxs = []
    second_idxs = []
    for idx in range(B):
      first, second = zip(*list(combinations(range(len(labels[idx])), 2)))
      first = torch.tensor(first)
      second = torch.tensor(second)
  
      pairwise_labels = (text_labels[first] != 0) & (image_labels[second] != 0) & \
                      (text_labels[first] == image_labels[second])    
      # first = [first_idx for first_idx in range(len(text_labels)) for second_idx in range(len(image_labels))]
      # second = [second_idx for first_idx in range(len(text_labels)) for second_idx in range(len(image_labels))]
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
          if config['loss'] == 'hinge':
            pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(-1, device=device))
          else:
            pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(0, device=device))

      pairwise_labels = pairwise_labels.to(torch.long).to(device)
      first_idxs.append(first)
      second_idxs.append(second)
      pairwise_labels_all.append(pairwise_labels)

    pairwise_labels_all = torch.stack(pairwise_labels_all)
    first_idxs = torch.stack(first_idxs_all)
    second_idxs = torch.stack(second_idxs_all)
    torch.cuda.empty_cache()

    return first_idxs, second_idxs, pairwise_labels_all    


def train(text_model, mention_model, image_model, coref_model, train_loader, test_loader, args):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(True)
  fix_seed(config)
  
  if not isinstance(text_model, torch.nn.DataParallel):
    text_model = nn.DataParallel(text_model)

  if not isinstance(text_model, torch.nn.DataParallel):
    mention_model = nn.DataParallel(mention_model)

  if not isinstance(image_model, torch.nn.DataParallel):
    image_model = nn.DataParallel(image_model)
  
  if not isinstance(coref_model, torch.nn.DataParallel):
    coref_model = nn.DataParallel(coref_model)

  text_model.to(device)
  # image_model.to(device)
  coref_model.to(device)

  # Create/load exp
  if not os.path.isdir(args.exp_dir):
    os.path.mkdir(args.exp_dir)

  if args.start_epoch != 0:
    text_model.load_state_dict(torch.load('{}/text_model.{}.pth'.format(args.exp_dir, args.start_epoch)))
    # image_model.load_state_dict(torch.load('{}/image_model.{}.pth'.format(args.exp_dir, args.start_epoch)))
    coref_model.load_state_dict(torch.load('{}/text_scorer.{}.pth'.format(args.exp_dir, args.start_epoch)))

  # Define the training criterion
  criterion = nn.BCEWithLogitsLoss()
  
  # Set up the optimizer  
  optimizer = get_optimizer(config, [text_model, image_model, coref_model])
   
  # Start training
  text_model.train()
  image_model.train()
  coref_model.train()
  total_loss = 0.
  total = 0.
  begin_time = time.time()
  if args.evaluate_only:
    config.epochs = 0
  for epoch in range(args.start_epoch, config.epochs):
    for i, batch in enumerate(train_loader):
      doc_embeddings,\
      start_mappings, end_mappings, continuous_mappings,\
      width, text_labels, text_mask, span_mask = batch   

      B = doc_embeddings.size(0)     
      doc_embeddings = doc_embeddings.to(device)
      start_mappings = start_mappings.to(device)
      end_mappings = end_mappings.to(device)
      continuous_mappings = continuous_mappings.to(device)
      width = width.to(device)
      # videos = videos.to(device)
      text_labels = text_labels.to(device).flatten()
      # img_labels = img_labels.to(device).flatten()
      span_mask = span_mask.to(device)
      span_num = span_mask.sum(-1).long()
      # video_mask = video_mask.to(device)
      # region_num = video_mask.sum(-1).long()
      first_idxs, second_idxs, pairwise_labels = get_pairwise_labels(text_labels, is_training=False, device=device)
      pairwise_labels = pairwise_labels.to(torch.float)
      if first_idxs is None:
        continue

      optimizer.zero_grad()

      text_output = text_model(doc_embeddings)
      start_embeddings = torch.matmul(start_mappings, text_output)
      end_embeddings = torch.matmul(end_mappings, text_output)
      start_end_embeddings = torch.cat([start_embeddings, end_embeddings], dim=-1)
      continuous_embeddings = torch.matmul(continuous_mappings, text_output.unsqueeze(1))
      mention_output = mention_model(start_end_embeddings, continuous_embeddings, width)
      # video_output = image_model(videos)
      scores = []
      for idx in range(B):
        scores.append(coref_model(mention_output[idx, first_idxs[idx]], 
                                  mention_output[idx, second_idxs[idx]]))
      scores = torch.stack(scores)
      if config.loss == 'mse':
        loss = - (pairwise_labels * scores).mean()
      else:
        loss = criterion(scores, pairwise_labels)       
      
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
      test(text_model, mention_model, image_model, coref_model, test_loader, args)
      
  if args.evaluate_only:
    test(text_model, mention_model, image_model, coref_model, test_loader, args)

    
def test(text_model, mention_model, coref_model, test_loader, args): # TODO Beam search
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = pyhocon.ConfigFactory.parse_file(args.config)
    documents = test_loader.dataset.documents
    all_scores = []
    all_labels = []
    text_model.eval()
    image_model.eval()
    coref_model.eval()
    with torch.no_grad(): 
      pred_dicts = []
      for i, batch in enumerate(test_loader):
        doc_embeddings,\
        start_mappings, end_mappings, continuous_mappings,\
        width, text_labels, text_mask, span_mask = batch   
        
        token_num = text_mask.sum(-1).long()
        span_num = span_mask.sum(-1).long()
        region_num = video_mask.sum(-1).long()
        doc_embeddings = doc_embeddings.to(device)
        start_mappings = start_mappings.to(device)
        end_mappings = end_mappings.to(device)
        continuous_mappings = continuous_mappings.to(device)
        width = width.to(device)
        text_labels = text_labels.to(device)
        
        # Compute mention embeddings
        text_output = text_model(doc_embeddings)
        start_embeddings = torch.matmul(start_mappings, text_output)
        end_embeddings = torch.matmul(end_mappings, text_output)
        start_end_embeddings = torch.cat([start_embeddings, end_embeddings], dim=-1)
        continuous_embeddings = torch.matmul(continuous_mappings, text_output.unsqueeze(1))
        mention_output = mention_model(start_end_embeddings, continuous_embeddings, width)

        B = doc_embeddings.size(0)  
        for idx in range(B):
          # Compute pairwise labels
          first_idx, second_idx, pairwise_labels = get_pairwise_labels(text_labels[idx, :span_num[idx]].unsqueeze(0), is_training=False, device=device)
          if first_idx is None:
            continue
          
          clusters, scores = coref_model.module.predict_cluster(text_output[idx],\
                              first_idx, second_idx) # TODO
          all_scores.append(scores.squeeze(1))
          all_labels.append(pairwise_labels.to(torch.int))
          global_idx = i * test_loader.batch_size + idx
          doc_id = test_loader.dataset.doc_ids[global_idx] 
          origin_tokens = [token[2] for token in test_loader.dataset.origin_tokens[global_idx]]
          candidate_start_ends = test_loader.dataset.origin_candidate_start_ends[global_idx]
          # print(doc_id, clusters.values(), candidate_start_ends.tolist())
          doc_name = doc_id
          document = {doc_id:test_loader.dataset.documents[doc_id]}
          write_output_file(document, clusters, [doc_id]*candidate_start_ends.shape[0],
                            candidate_start_ends[:, 0].tolist(),
                            candidate_start_ends[:, 1].tolist(),
                            os.path.join(config['model_path'], 'pred_conll'),
                            doc_name, False, True)
                
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

if __name__ == '__main__':
  # Set up argument parser
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', type=str, default='')
  parser.add_argument('--config', type=str, default='configs/config_translation.json')
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
  
  pred_out_dir = os.path.join(config['model_path'], 'pred_conll')
  if not os.path.isdir(pred_out_dir):
    os.mkdir(pred_out_dir)

  logging.basicConfig(filename=os.path.join(config['log_path'],'{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                      format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) 

  # Initialize dataloaders
  train_set = TextFeatureDataset(os.path.join(config['data_folder'], 'train.json'), os.path.join(config['data_folder'], 'train_mixed.json'), config, split='train') # TODO
  test_set = TextFeatureDataset(os.path.join(config['data_folder'], 'test.json'), os.path.join(config['data_folder'], 'test_mixed.json'), config, split='test') # TODO
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

  # Initialize models
  text_model = NoOp() # BiLSTM(embedding_dim, embedding_dim)
  mention_model = SpanEmbedder(config, device).to(device)
  embedding_dim = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2
  image_model = NoOp()
  coref_model = SimplePairWiseClassifier(config).to(device)
  # coref_model = SelfAttentionPairWiseClassifier(config).to(device) # TODO

  if config['training_method'] in ('pipeline', 'continue'):
      text_model.load_state_dict(torch.load(config['span_repr_path'], map_location=device))
      coref_model.text_scorer.load_state_dict(torch.load(config['pairwise_scorer_path'], map_location=device))
  
  # Training
  train(text_model, mention_model, image_model, coref_model, train_loader, test_loader, args)
