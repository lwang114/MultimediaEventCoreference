import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import logging
import time
from datetime import datetime
import json
import argparse
import pyhocon 
import random
import numpy as np
from copy import deepcopy
from itertools import combinations
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from text_models import BiLSTM, SpanEmbedder, SimplePairWiseClassifier
from visual_models import BiLSTMVideoEncoder, CrossmediaPairWiseClassifier, ClassAttender
from corpus import TextVideoEventDataset
from evaluator import Evaluation, CoNLLEvaluation
from utils import create_type_stoi, create_feature_stoi


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

def get_pairwise_labels(text_labels, 
                        action_labels, 
                        is_training, 
                        device):
    B = text_labels.size(0)
    pairwise_labels = []
    first = [first_idx for first_idx in range(len(text_labels)) for second_idx in range(len(action_labels))]
    second = [second_idx for first_idx in range(len(text_labels)) for second_idx in range(len(action_labels))]
    pairwise_labels = (text_labels[first] == action_labels[second]).to(torch.long).to(device)      

    if config['loss'] == 'hinge' and is_training:
      pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(-1, device=device)) 

    return first, second, pairwise_labels

def get_pairwise_text_labels(labels, is_training, device):
    B = labels.size(0)
    pairwise_labels_all = []
    first_idxs = []
    second_idxs = []
    for idx in range(B):
      if len(labels[idx]) <= 1:
        return None, None, None
      first, second = zip(*list(combinations(range(len(labels[idx])), 2)))
      pairwise_labels = (labels[idx, first] != 0) & (labels[idx, second] != 0) & \
                      (labels[idx, first] == labels[idx, second])    
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
    first_idxs = np.stack(first_idxs)
    second_idxs = np.stack(second_idxs)
    torch.cuda.empty_cache()

    return first_idxs, second_idxs, pairwise_labels_all    

def train(text_model,
          mention_model, 
          visual_model, 
          attention_model,
          text_coref_model, 
          visual_coref_model,
          train_loader,
          test_loader, 
          args, random_seed=2):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(True)

  config = pyhocon.ConfigFactory.parse_file(args.config)
  weight_visual = config.weight_visual
  n_event_class = len(train_loader.dataset.event_stoi)
  if random_seed:
    config.random_seed = random_seed

  if not isinstance(text_model, torch.nn.DataParallel):
    text_model = nn.DataParallel(text_model)

  if not isinstance(mention_model, torch.nn.DataParallel):
    mention_model = nn.DataParallel(mention_model)

  if not isinstance(visual_model, torch.nn.DataParallel):
    visual_model = nn.DataParallel(visual_model)
  
  if not isinstance(attention_model, torch.nn.DataParallel):
    attention_model = nn.DataParallel(attention_model)

  text_model.to(device)
  mention_model.to(device)
  visual_model.to(device)
  attention_model.to(device)

  # Define the training criterion
  # criterion = nn.CrossEntropyLoss(ignore_index=0)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = get_optimizer(config, [text_model,
                                     mention_model, 
                                     visual_model,
                                     attention_model])
 
  # Start training
  total_loss = 0.
  total = 0.
  best_text_f1 = 0.
  best_event_f1 = 0.
  results = dict()
  begin_time = time.time()
  if args.evaluate_only:
    config.epochs = 0

  for epoch in range(config.epochs):
    text_model.train()
    mention_model.train()
    visual_model.train()
    attention_model.train()

    for i, batch in enumerate(train_loader):
      doc_embeddings = batch['doc_embeddings'].to(device)
      start_mappings = batch['start_mappings'].to(device)
      end_mappings = batch['end_mappings'].to(device)
      continuous_mappings = batch['continuous_mappings'].to(device)
      width = batch['width'].to(device)

      videos = batch['visual_embeddings'].to(device)
      text_labels = batch['cluster_labels'].to(device)
      event_labels = batch['event_labels'].to(device)

      text_mask = batch['text_mask'].to(device)
      span_mask = batch['span_mask'].to(device)
      action_span_mask = batch['visual_mask'].to(device)
      span_num = (span_mask.sum(-1) > 0).long().sum(-1)
      action_num = (action_span_mask.sum(-1) > 0).long().sum(-1)
      action_mask = (action_span_mask.sum(-1) > 0).float()
      mention_mask = (span_mask.sum(-1) > 0).float()
      B = doc_embeddings.size(0)
      
      # Compute mention embeddings
      mention_output = mention_model(doc_embeddings,
                                     start_mappings,
                                     end_mappings,
                                     continuous_mappings, 
                                     width, 
                                     attention_mask=text_mask)

      # Compute visual event logits of size (batch size, n event class)
      video_output = visual_model(videos, action_span_mask)
      # (batch size, action num, n event class)
      visual_event_logit, _ = attention_model(video_output, action_mask)
      
      # Compute textual event logits of size (batch size, span num, n event class)
      text_output = text_model(mention_output)
      text_event_logit, _ = attention_model(text_output, mention_mask) 

      # text_event_logits = weight_visual * visual_event_logit.unsqueeze(-2) + (1 - weight_visual) * text_event_logits
      # loss = criterion(text_event_logits.view(-1, n_event_class), event_labels.flatten())
      binary_event_labels = ((F.one_hot(event_labels, n_event_class) * mention_mask.unsqueeze(-1)).sum(-2) > 0).float()
      text_event_logit = weight_visual * visual_event_logit + (1 - weight_visual) * text_event_logit
      loss = criterion(text_event_logit, binary_event_labels)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item() * B
      total += B
      if i % 200 == 0:
        info = f'Iter {i} {total_loss / total:.4f}'
        print(info)
        logging.info(info)
    
    info = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tTime {time.time()-begin_time:.3f}\tLoss total {total_loss:.4f} ({total_loss / total:.4f})'
    print(info)
    logging.info(info)
    torch.save(visual_model.module.state_dict(), '{}/visual_model-{}.pth'.format(config['model_path'], random_seed))
    torch.save(text_model.module.state_dict(), '{}/text_model-{}.pth'.format(config['model_path'], random_seed))
    torch.save(mention_model.module.state_dict(), '{}/mention_model-{}.pth'.format(config['model_path'], random_seed))
    torch.save(attention_model.module.state_dict(), '{}/attention_model-{}.pth'.format(config['model_path'], random_seed))
    
    if epoch % 1 == 0:
      res = test(text_model,
                 mention_model,
                 visual_model,
                 attention_model,
                 text_coref_model,
                 visual_coref_model,
                 test_loader, args,
                 epoch=epoch,
                 random_seed=random_seed)
      if (len(results) == 0) or (res['event'][-1] > best_event_f1):
        best_event_f1 = res['event'][-1]
        results = deepcopy(res)
        torch.save(text_model.module.state_dict(), '{}/best_text_model-{}.pth'.format(config['model_path'], random_seed))
        torch.save(mention_model.module.state_dict(), '{}/best_mention_model-{}.pth'.format(config['model_path'], random_seed))
        torch.save(visual_model.module.state_dict(), '{}/best_visual_model-{}.pth'.format(config['model_path'], random_seed))
        torch.save(attention_model.module.state_dict(), '{}/best_attention_model-{}.pth'.format(config['model_path'], random_seed))
      
  if args.evaluate_only:
    results = test(text_model,
                   mention_model,
                   visual_model,
                   attention_model,
                   text_coref_model,
                   visual_coref_model,
                   test_loader,
                   args,
                   random_seed=random_seed)
  return results

def test(text_model,
         mention_model,
         visual_model,
         attention_model,
         text_coref_model,
         visual_coref_model,
         test_loader,
         args,
         epoch=0,
         random_seed=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = pyhocon.ConfigFactory.parse_file(args.config)
    weight_visual = config.weight_visual
    n_event_class = len(test_loader.dataset.event_stoi)
    event_types = sorted(test_loader.dataset.event_stoi, key=lambda x:test_loader.dataset.event_stoi[x])
    
    pred_event_labels = []
    gold_event_labels = []
    
    text_model.eval()
    mention_model.eval()
    visual_model.eval()
    attention_model.eval()

    conll_eval = CoNLLEvaluation()
    f_out = open(os.path.join(config['model_path'], f'prediction_{random_seed}_{epoch}.readable'), 'w')
    best_f1 = 0.
    results = dict()
    with torch.no_grad():
      for i, batch in enumerate(test_loader):
        doc_embeddings = batch['doc_embeddings'].to(device)
        start_mappings = batch['start_mappings'].to(device)
        end_mappings = batch['end_mappings'].to(device)
        continuous_mappings = batch['continuous_mappings'].to(device)
        width = batch['width'].to(device)
        
        videos = batch['visual_embeddings'].to(device)
        text_labels = batch['cluster_labels'].to(device) 
        event_labels = batch['event_labels'].to(device)
        text_mask = batch['text_mask'].to(device)
        span_mask = batch['span_mask'].to(device)
        action_span_mask = batch['visual_mask'].to(device)
        span_num = (span_mask.sum(-1) > 0).long().sum(-1)
        action_num = (action_span_mask.sum(-1) > 0).long().sum(-1)
        action_mask = (action_span_mask.sum(-1) > 0).float()
        mention_mask = (span_mask.sum(-1) > 0).float()
  
        # Compute mention embeddings
        mention_output = mention_model(doc_embeddings,
                                       start_mappings,
                                       end_mappings,
                                       continuous_mappings, 
                                       width, 
                                       attention_mask=text_mask)

        # Compute visual event logits of size (batch size, n event class)
        video_output = visual_model(videos, action_span_mask)
        # (batch size, action num, n event class)
        visual_event_logit, _ = attention_model(video_output, action_mask)
        
        # Compute textual event logits of size (batch size, span num, n event class)
        text_output = text_model(mention_output)
        text_event_logit, _ = attention_model(text_output, mention_mask) 
        
        # text_event_probs = F.softmax(text_event_logits, dim=-1)
        # text_event_logits = weight_visual * visual_event_logit.unsqueeze(-2) + (1 - weight_visual) * text_event_logits

        binary_event_labels = ((F.one_hot(event_labels, n_event_class) * mention_mask.unsqueeze(-1)).sum(-2) > 0).float()
        text_event_logit = weight_visual * visual_event_logit + (1 - weight_visual) * text_event_logit
        
        B = doc_embeddings.size(0)
        for idx in range(B):
          global_idx = i * test_loader.batch_size + idx
          if span_num[idx] == 0:
              continue
          doc_id = test_loader.dataset.doc_ids[global_idx]
          tokens = [token[2] for token in test_loader.dataset.documents[doc_id]]
          # pred_event_label = text_event_probs[idx, :span_num[idx]].max(-1)[1].cpu().detach()
          pred_event_label = (text_event_logit[idx] > 0).long().cpu().detach()
          # gold_event_label = event_labels[idx, :span_num[idx]].cpu().detach()
          gold_event_label = binary_event_labels[idx].long().cpu().detach()
          
          event_label_dict = test_loader.dataset.event_label_dict[doc_id]
          mention_str = ', '.join([' '.join(tokens[start:end+1]) for start, end in sorted(event_label_dict)])
          # pred_event_str = ', '.join([event_types[p] for p in pred_event_label.numpy()])
          # gold_event_str = ', '.join([event_types[g] for g in gold_event_label.numpy()])
          pred_event_str = ', '.join([event_types[p] for p in range(n_event_class) if pred_event_label.numpy()[p] > 0])
          gold_event_str = ', '.join([event_types[g] for g in range(n_event_class) if gold_event_label.numpy()[g] > 0])
          token_str = ' '.join(tokens).replace('\n', '')
          f_out.write(f"{doc_id}: {token_str}\n")
          f_out.write(f'Mentions: {mention_str}\n')
          f_out.write(f'Pred Events: {pred_event_str}\n')
          f_out.write(f'Gold Events: {gold_event_str}\n\n')
          
          pred_event_labels.append(pred_event_label)
          gold_event_labels.append(gold_event_label)
      f_out.close()
      pred_event_labels = torch.cat(pred_event_labels)
      gold_event_labels = torch.cat(gold_event_labels)
      # event_prec, event_rec, event_f1, _ = precision_recall_fscore_support(gold_event_labels.numpy(),
      #                                                                     pred_event_labels.numpy(), average='macro')
      event_prec, event_rec, event_f1, _ = precision_recall_fscore_support(gold_event_labels.numpy(),
                                                                           pred_event_labels.numpy(), average='binary')
      print('[Event Classification Result]')
      info = f'Event classification - Recall: {event_prec:.4f}, Precision: {event_rec:.4f}, F1: {event_f1:.4f}\n'
      print(info)
      logging.info(info)

      results['event'] = [event_prec, event_rec, event_f1]
      return results

if __name__ == '__main__':
  # Set up argument parser
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='configs/config_multimodal_coref_video_m2e2.json')
  parser.add_argument('--start_epoch', type=int, default=0)
  parser.add_argument('--evaluate_only', action='store_true')
  args = parser.parse_args()
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  config = pyhocon.ConfigFactory.parse_file(args.config)
  print(config['model_path'])
  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  if not os.path.isdir(os.path.join(config['model_path'], 'log')):
    os.mkdir(os.path.join(config['model_path'], 'log')) 

  # Set up logger
  logging.basicConfig(filename=os.path.join(config['model_path'],'log/{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                      format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) 

  # Initialize dataloaders
  splits = [os.path.join(config['data_folder'], 'train_events.json'),
            os.path.join(config['data_folder'], 'test_events.json')]
  # os.path.join(config['data_folder'], 'train_unlabeled_events.json'
  
  event_stoi = create_type_stoi(splits) 
  json.dump(event_stoi, open('event_stoi', 'w'), indent=2) # XXX
  
  train_set = TextVideoEventDataset(config, 
                                    event_stoi, 
                                    dict(),
                                    splits=config.splits['train'])
  test_set = TextVideoEventDataset(config, 
                                   event_stoi, 
                                   dict(),
                                   splits=config.splits['test'])

  event_metrics = []
  
  seeds = config.seeds
  for seed in seeds:
      config.random_seed = seed
      config['random_seed'] = seed
      fix_seed(config)
  
      # Initialize dataloaders 
      train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
      test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

      # Initialize models
      mention_model = SpanEmbedder(config, device)
      text_coref_model = SimplePairWiseClassifier(config).to(device)
      text_model = nn.Sequential(
                     nn.Linear(int(text_coref_model.input_layer // 3), config.hidden_layer),
                     nn.ReLU(),
                     nn.Linear(config.hidden_layer, config.hidden_layer),
                     nn.ReLU(),
                     nn.Linear(config.hidden_layer, config.hidden_layer),
                     nn.ReLU())
      attention_model = ClassAttender(config.hidden_layer,
                                      config.hidden_layer,
                                      len(event_stoi))
      visual_model = BiLSTMVideoEncoder(400, int(config.hidden_layer // 2))
      visual_coref_model = CrossmediaPairWiseClassifier(config).to(device)
      
      if config['training_method'] in ('pipeline', 'continue') or args.evaluate_only:
        text_model.load_state_dict(torch.load(config['text_model_path'], map_location=device))
        mention_model.load_state_dict(torch.load(config['mention_model_path']))
        attention_model.load_state_dict(torch.load(config['attention_model_path']))
      
      n_params = 0
      for p in text_model.parameters():
          n_params += p.numel()

      for p in mention_model.parameters():
          n_params += p.numel()

      for p in text_coref_model.parameters():
          n_params += p.numel()
    
      for p in attention_model.parameters():
          n_params += p.numel()
          
      for p in visual_coref_model.parameters():
          n_params += p.numel()

      print('Number of parameters in coref classifier: {}'.format(n_params))
      results = train(text_model,
                      mention_model,
                      visual_model,
                      attention_model,                      
                      text_coref_model, 
                      visual_coref_model,
                      train_loader, 
                      test_loader, 
                      args, random_seed=seed)
      event_metrics.append(results['event'])

  mean_event, std_event = np.mean(np.asarray(event_metrics), axis=0), np.std(np.asarray(event_metrics), axis=0)

  info = f'Event: precision {mean_event[0]} +/- {std_event[0]}, '\
         f'recall {mean_event[1]} +/- {std_event[1]}'\
         f'f1 {mean_event[2]} +/- {std_event[2]}'
  print(info)
  logging.info(info) 
