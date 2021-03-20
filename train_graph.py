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
from text_models import SpanEmbedder, StarTransformerClassifier
from image_models import VisualEncoder
from criterion import TripletLoss
from corpus_graph import StarFeatureDataset
from evaluator import Evaluation, RetrievalEvaluation, CoNLLEvaluation
from utils import make_prediction_readable
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

def get_pairwise_labels(text_labels_batch, image_labels_batch, is_training, device):
    B = text_labels_batch.size(0)
    first_batch = []
    second_batch = []
    pairwise_labels_batch = []
    for idx in range(B):
      image_labels = image_labels_batch[idx]
      text_labels = text_labels_batch[idx]
      first = [first_idx for first_idx in range(len(text_labels)) for second_idx in range(len(image_labels))]
      second = [second_idx for first_idx in range(len(text_labels)) for second_idx in range(len(image_labels))]
      first = torch.tensor(first)
      second = torch.tensor(second)
      pairwise_labels = (text_labels[first] != 0) & (image_labels[second] != 0) & \
                        (text_labels[first] == image_labels[second])
      pairwise_labels = pairwise_labels.to(torch.long).to(device)      

      if config['loss'] == 'hinge' and is_training:
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(-1, device=device)) 
      else:
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(0, device=device))
      # torch.cuda.empty_cache()
      first_batch.append(first)
      second_batch.append(second)
      pairwise_labels_batch.append(pairwise_labels)

    first_batch = torch.stack(first_batch, dim=0)
    second_batch = torch.stack(second_batch, dim=0)
    pairwise_labels_batch = torch.stack(pairwise_labels_batch, dim=0)
    return first_batch, second_batch, pairwise_labels_batch

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

def train(text_model, mention_model, image_model, coref_model, train_loader, test_loader, args):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(True)
  fix_seed(config)
  
  if not isinstance(text_model, torch.nn.DataParallel):
    text_model = nn.DataParallel(text_model)

  if not isinstance(mention_model, torch.nn.DataParallel):
    mention_model = nn.DataParallel(mention_model)

  if not isinstance(image_model, torch.nn.DataParallel):
    image_model = nn.DataParallel(image_model)
  
  if not isinstance(coref_model, torch.nn.DataParallel):
    coref_model = nn.DataParallel(coref_model)

  text_model.to(device)
  mention_model.to(device)
  image_model.to(device)
  coref_model.to(device)

  # Create/load exp
  if not os.path.isdir(args.exp_dir):
    os.path.mkdir(args.exp_dir)

  if args.start_epoch != 0:
    text_model.load_state_dict(torch.load('{}/text_model.{}.pth'.format(args.exp_dir, args.start_epoch)))
    image_model.load_state_dict(torch.load('{}/image_model.{}.pth'.format(args.exp_dir, args.start_epoch)))
    coref_model.load_state_dict(torch.load('{}/text_scorer.{}.pth'.format(args.exp_dir, args.start_epoch)))

  # Define the training criterion
  criterion = nn.BCEWithLogitsLoss()
  multimedia_criterion = TripletLoss(config) 

  # Set up the optimizer  
  optimizer = get_optimizer(config, [text_model, image_model, coref_model])
   
  # Start training
  total_loss = 0.
  total = 0.
  best_text_f1 = 0.
  best_grounding_f1 = 0.
  best_retrieval_recall = 0.
  begin_time = time.time()
  if args.evaluate_only:
    config.epochs = 0
  for epoch in range(args.start_epoch, config.epochs):
    text_model.train()
    mention_model.train()
    image_model.train()
    coref_model.train()
    for i, batch in enumerate(train_loader):
      doc_embeddings,\
      start_mappings,\
      end_mappings,\
      continuous_mappings,\
      event_mappings, entity_mappings,\
      event_to_roles_mappings,\
      width, videos,\
      text_labels, type_labels,\
      img_labels,\
      text_mask, span_mask, video_mask = batch   

      B = doc_embeddings.size(0)     
      doc_embeddings = doc_embeddings.to(device)
      start_mappings = start_mappings.to(device)
      end_mappings = end_mappings.to(device)
      continuous_mappings = continuous_mappings.to(device)
      event_mappings = event_mappings.to(device)
      entity_mappings = entity_mappings.to(device)
      event_to_roles_mappings = event_to_roles_mappings.to(device)

      width = width.to(device)
      videos = videos.to(device)
      text_labels = text_labels.to(device)
      type_labels = type_labels.to(device)
      img_labels = img_labels.to(device)
      text_mask = text_mask.to(device)
      span_mask = span_mask.to(device)
      video_mask = video_mask.to(device)
      
      first_grounding_idx, second_grounding_idx, pairwise_grounding_labels = get_pairwise_labels(text_labels, img_labels, is_training=False, device=device)      
      event_labels = torch.bmm(event_mappings, text_labels.unsqueeze(-1).float()).squeeze(-1).long()
      entity_labels = torch.bmm(entity_mappings, text_labels.unsqueeze(-1).float()).squeeze(-1).long()
      first_event_idx, second_event_idx, pairwise_event_labels = get_pairwise_text_labels(event_labels, is_training=False, device=device)
      first_entity_idx, second_entity_idx, pairwise_entity_labels = get_pairwise_text_labels(entity_labels, is_training=False, device=device)
 
      pairwise_grounding_labels = pairwise_grounding_labels.to(torch.float)
      pairwise_event_labels = pairwise_event_labels.to(torch.float).flatten()
      pairwise_entity_labels = pairwise_entity_labels.to(torch.float).flatten()

      optimizer.zero_grad()

      text_output = text_model(doc_embeddings)
      video_output = image_model(videos)
      mention_output = mention_model(text_output,
                                     start_mappings, end_mappings,
                                     continuous_mappings, width,
                                     type_labels=type_labels)

      '''
      event_scores, entity_scores = [], []
      for idx in range(B):
          event_scores.append(coref_model(mention_output[idx].unsqueeze(0),
                                          event_mappings[idx].unsqueeze(0), 
                                          event_to_roles_mappings[idx].unsqueeze(0),
                                          first_event_idx[idx],
                                          second_event_idx[idx]).squeeze(0))

      for idx in range(B):
          entity_scores.append(coref_model(mention_output[idx].unsqueeze(0),
                                           entity_mappings[idx].unsqueeze(0),
                                           entity_mappings[idx].unsqueeze(1).unsqueeze(0),
                                           first_entity_idx[idx],
                                           second_entity_idx[idx]).squeeze(0))
      event_scores = torch.cat(event_scores)
      entity_scores = torch.cat(entity_scores)
      '''
      if first_grounding_idx is None or (first_event_idx is None and first_entity_idx is None):
        continue

      if not first_event_idx is None:
          event_scores = coref_model(mention_output,
                                 event_mappings,
                                 event_to_roles_mappings,
                                 first_event_idx[0],
                                 second_event_idx[0]).flatten()
          loss_event = criterion(event_scores, pairwise_event_labels)
      else:
          loss_event = torch.zeros((mention_output.size(0),), dtype=torch.float, device=device)
          
      if not first_entity_idx is None:
          entity_scores = coref_model(mention_output,
                                  entity_mappings,
                                  entity_mappings.unsqueeze(-2),
                                  first_entity_idx[0],
                                  second_entity_idx[0]).flatten()
          loss_entity = criterion(entity_scores, pairwise_entity_labels)
      else:
          loss_entity = torch.zeros((mention_output.size(0),), dtype=torch.float, device=device)
          
      loss = loss_event + loss_entity +  multimedia_criterion(text_output, video_output,
                                                              text_mask, video_mask)

      loss.backward()
      optimizer.step()

      total_loss += loss.item() * B
      total += B
      if i % 200 == 0:
        info = 'Iter {} {:.4f}'.format(i, total_loss / total)
        print(info)
        logger.info(info) 
    
    info = 'Epoch: [{}][{}/{}]\tTime {:.3f}\tLoss total {:.4f} ({:.4f})'.format(epoch, i, len(train_loader), time.time()-begin_time, total_loss, total_loss / total)
    print(info)
    logger.info(info)

    torch.save(text_model.module.state_dict(), '{}/text_model.pth'.format(args.exp_dir))
    torch.save(image_model.module.state_dict(), '{}/image_model.pth'.format(args.exp_dir))
    torch.save(mention_model.module.state_dict(), '{}/mention_model.pth'.format(args.exp_dir))
    torch.save(coref_model.module.state_dict(), '{}/text_scorer.pth'.format(args.exp_dir))
 
    if epoch % 1 == 0:
      task = config.get('task', 'coreference')
      if task in ('coreference', 'both'):
        text_f1 = test(text_model, mention_model, image_model, coref_model, test_loader, args)
        if text_f1 > best_text_f1:
          best_text_f1 = text_f1
          torch.save(text_model.module.state_dict(), '{}/best_text_model.pth'.format(args.exp_dir))
          torch.save(image_model.module.state_dict(), '{}/best_image_model.pth'.format(args.exp_dir))
          torch.save(mention_model.module.state_dict(), '{}/best_mention_model.pth'.format(args.exp_dir))
          torch.save(coref_model.module.state_dict(), '{}/best_text_scorer.pth'.format(args.exp_dir))
        print('Best text coreference F1={}'.format(best_text_f1))
      if task in ('retrieval', 'both'):
        I2S_r10, S2I_r10 = test_retrieve(text_model, image_model, grounding_model, test_loader, args)
        if (I2S_r10 + S2I_r10) / 2 > best_retrieval_recall:
          best_retrieval_recall = (I2S_r10 + S2I_r10) / 2 
        print('Best avg. Recall@10={}'.format((I2S_r10 + S2I_r10) / 2))

  if args.evaluate_only:
    task = config.get('task', 'coreference')
    print(task)
    task = config.get('task', 'coreference')
    if task in ('coreference', 'both'):
      text_f1 = test(text_model, mention_model, image_model, coref_model, test_loader, args)
    if task in ('retrieval', 'both'):
      I2S_r10, S2I_r10 = test_retrieve(text_model, image_model, test_loader, args)


def test(text_model, mention_model, image_model, coref_model, test_loader, args): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = pyhocon.ConfigFactory.parse_file(args.config)
    documents = test_loader.dataset.documents
    all_event_scores, all_entity_scores = [], []
    all_event_labels, all_entity_labels = [], []
    text_model.eval()
    image_model.eval()
    coref_model.eval()

    conll_eval_event = CoNLLEvaluation()
    conll_eval_entity = CoNLLEvaluation()
    conll_eval = CoNLLEvaluation()
    best_f1 = 0. 
    f_out = open(os.path.join(args.exp_dir, 'prediction.readable'), 'w')
    with torch.no_grad():
        pred_dicts = []      
        for i, batch in enumerate(test_loader):
          doc_embeddings,\
          start_mappings, end_mappings,\
          continuous_mappings,\
          event_mappings, entity_mappings,\
          event_to_roles_mappings,\
          width, videos,\
          text_labels, type_labels,\
          img_labels,\
          text_mask, span_mask, video_mask = batch

          token_num = text_mask.sum(-1).long()
          region_num = video_mask.sum(-1).long()
          doc_embeddings = doc_embeddings.to(device)
          start_mappings = start_mappings.to(device)
          end_mappings = end_mappings.to(device)
          continuous_mappings = continuous_mappings.to(device)
          event_mappings = event_mappings.to(device)
          entity_mappings = entity_mappings.to(device)
          event_to_roles_mappings = event_to_roles_mappings.to(device)

          width = width.to(device)
          videos = videos.to(device)
          text_labels = text_labels.to(device)
          img_labels = img_labels.to(device)
          text_mask = text_mask.to(device)
          span_mask = span_mask.to(device)
          video_mask = video_mask.to(device)

          # Extract span and video embeddings
          text_output = text_model(doc_embeddings)
          video_output = image_model(videos)
          mention_output = mention_model(text_output,
                                         start_mappings, end_mappings,
                                         continuous_mappings, width,
                                         type_labels=type_labels)

          event_labels = torch.bmm(event_mappings, text_labels.float().unsqueeze(-1)).squeeze(-1).long()
          entity_labels = torch.bmm(entity_mappings, text_labels.float().unsqueeze(-1)).squeeze(-1).long()
          event_num = event_mappings.sum(-1).sum(-1).long()
          entity_num = entity_mappings.sum(-1).sum(-1).long()
          
          # Compute score for each mention pair
          B = doc_embeddings.size(0) 
          for idx in range(B):
            global_idx = i * test_loader.batch_size + idx
            
            # Compute pairwise labels
            first_event_idx, second_event_idx, pairwise_event_labels = get_pairwise_text_labels(event_labels[idx, :event_num[idx]].unsqueeze(0), 
                                                                                             is_training=False, device=device)
            first_entity_idx, second_entity_idx, pairwise_entity_labels = get_pairwise_text_labels(entity_labels[idx, :entity_num[idx]].unsqueeze(0),
                                                                                                   is_training=False, device=device)
            if first_event_idx is None and first_entity_idx is None:
                continue

            origin_candidate_start_ends = torch.LongTensor(test_loader.dataset.origin_candidate_start_ends[global_idx])
            mention_num = origin_candidate_start_ends.shape[0]
            if not first_event_idx is None:
                first_event_idx = first_event_idx[0]
                second_event_idx = second_event_idx[0]
                pairwise_event_labels = pairwise_event_labels.squeeze(0)
                event_mapping = event_mappings[idx, :event_num[idx]]
                
                event_start_ends = torch.mm(event_mapping[:, :mention_num].cpu(), origin_candidate_start_ends.float()).long()
                event_to_roles_mapping = event_to_roles_mappings[idx, :event_num[idx]]
                event_label = event_labels[idx, :event_num[idx]]
                # Compute pairwise scores
                event_antecedents, event_scores = coref_model.module.predict_cluster(mention_output[idx],
                                                                                 event_mapping, event_to_roles_mapping,
                                                                                 first_event_idx, second_event_idx)
                event_antecedents = torch.LongTensor(event_antecedents)
                all_event_scores.append(event_scores)
                all_event_labels.append(pairwise_event_labels.to(torch.int).cpu())
                pred_event_clusters, gold_event_clusters = conll_eval(event_start_ends,
                                                                      event_antecedents,
                                                                      event_start_ends,
                                                                      event_label)
                _, _ = conll_eval_event(event_start_ends,
                                        event_antecedents,
                                        event_start_ends,
                                        event_label)
            else:
                pred_event_clusters, gold_event_clusters = [], []
                
            if not first_entity_idx is None:
                first_entity_idx = first_entity_idx[0]
                second_entity_idx = second_entity_idx[0]
                pairwise_entity_labels = pairwise_entity_labels.squeeze(0)
                entity_mapping = entity_mappings[idx, :entity_num[idx]]
                entity_start_ends = torch.mm(entity_mapping[:, :mention_num].cpu(), origin_candidate_start_ends.float()).long()
                entity_label = entity_labels[idx, :entity_num[idx]]
                entity_antecedents, entity_scores = coref_model.module.predict_cluster(mention_output[idx],
                                                                                   entity_mapping, entity_mapping.unsqueeze(1),
                                                                                   first_entity_idx, second_entity_idx)
                entity_antecedents = torch.LongTensor(entity_antecedents)
                all_entity_scores.append(entity_scores)
                all_entity_labels.append(pairwise_entity_labels.to(torch.int).cpu())

                pred_entity_clusters, gold_entity_clusters = conll_eval(entity_start_ends,
                                                                        entity_antecedents,
                                                                        entity_start_ends,
                                                                        entity_label) 
                _, _ = conll_eval_entity(entity_start_ends,
                                         entity_antecedents,
                                         entity_start_ends,
                                         entity_label)
            else:
                pred_entity_clusters, gold_entity_clusters = [], []

            # Save the output clusters
            doc_id = test_loader.dataset.doc_ids[global_idx]
            tokens = [token[2] for token in test_loader.dataset.documents[doc_id]]

            pred_clusters_str, gold_clusters_str = conll_eval.make_output_readable(pred_event_clusters+pred_entity_clusters,
                                                                                   gold_event_clusters+gold_entity_clusters,
                                                                                   tokens)
            token_str = ' '.join(tokens).replace('\n', '')
            f_out.write(f"{doc_id}: {token_str}\n")
            f_out.write(f'Pred: {pred_clusters_str}\n')
            f_out.write(f'Gold: {gold_clusters_str}\n\n')
 
        all_scores = torch.cat(all_event_scores+all_entity_scores)
        all_labels = torch.cat(all_event_labels+all_entity_labels)
        all_event_scores = torch.cat(all_event_scores)
        all_event_labels = torch.cat(all_event_labels)
        all_entity_scores = torch.cat(all_entity_scores)
        all_entity_labels = torch.cat(all_entity_labels)
        
        strict_preds = (all_scores > 0).to(torch.int) 
        strict_preds_event = (all_event_scores > 0).to(torch.int)
        strict_preds_entity = (all_entity_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))
        eval_event = Evaluation(strict_preds_event, all_event_labels.to(device))
        eval_entity = Evaluation(strict_preds_entity, all_entity_labels.to(device))
        print('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        print('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                     len(all_labels)))
        print('Pairwise - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                eval.get_precision(), eval.get_f1()))
        muc, b_cubed, ceafe, avg = conll_eval.get_metrics()
        conll_metrics = muc+b_cubed+ceafe+avg
        print('MUC - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'Bcubed - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CEAFe - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CoNLL - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(*conll_metrics)) 
        
        print('Event Pairwise - Recall: {}, Precision: {}, F1: {}'.format(eval_event.get_recall(),
                                                                eval_event.get_precision(), eval_event.get_f1()))
        muc, b_cubed, ceafe, avg = conll_eval_event.get_metrics()
        conll_metrics = muc+b_cubed+ceafe+avg
        print('Event MUC - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'Event Bcubed - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'Event CEAFe - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'Event CoNLL - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(*conll_metrics))
        
        logger.info('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        logger.info('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                           len(all_labels)))
        logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                      eval.get_precision(), eval.get_f1()))
        return eval.get_f1()


def test_retrieve(text_model, image_model, grounding_model, test_loader, args):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  text_embeddings = []
  video_embeddings = []
  text_masks = []
  video_masks = []
  text_model.eval()
  image_model.eval()
  criterion = TripletLoss()
  
  with torch.no_grad():
    pred_dicts = []
    for i, batch in enumerate(test_loader):
      doc_embeddings, start_mappings, end_mappings, continuous_mappings,\
      width, videos,\
      text_labels, img_labels,\
      text_mask, span_mask, video_mask = batch   

      token_num = text_mask.sum(-1).long()
      span_num = torch.where(span_mask.sum(-1) > 0, 
                             torch.tensor(1, dtype=torch.int, device=doc_embeddings.device), 
                             torch.tensor(0, dtype=torch.int, device=doc_embeddings.device)).sum(-1)

      region_num = video_mask.sum(-1).long()
      doc_embeddings = doc_embeddings.to(device)
      start_mappings = start_mappings.to(device)
      end_mappings = end_mappings.to(device)
      continuous_mappings = continuous_mappings.to(device)
      width = width.to(device)
      videos = videos.to(device)
      text_labels = text_labels.to(device)
      img_labels = img_labels.to(device)
      span_mask = span_mask.to(device)
      video_mask = video_mask.to(device)

      # Extract span and video embeddings
      text_output = text_model(doc_embeddings)
      video_output = image_model(videos)

      text_output = text_output.cpu().detach()
      video_output = video_output.cpu().detach()
      span_mask = span_mask.cpu().detach()
      video_mask = video_mask.cpu().detach()

      text_embeddings.extend(torch.split(text_output, 1))
      video_embeddings.extend(torch.split(video_output, 1))
      text_masks.extend(torch.split(text_mask, 1))
      video_masks.extend(torch.split(video_mask, 1))

    I2S_idxs, S2I_idxs = criterion.retrieve(text_embeddings, video_embeddings, text_masks, video_masks, k=10)
    with open(os.path.join(args.exp_dir, 'I2S.txt'), 'w') as f_i2s,\
         open(os.path.join(args.exp_dir, 'S2I.txt'), 'w') as f_s2i:
      for idx, (i2s_idxs, s2i_idxs) in enumerate(zip(I2S_idxs.cpu().detach().numpy().tolist(), S2I_idxs.cpu().detach().numpy().tolist())):
        doc_id = test_loader.dataset.doc_ids[idx]
        tokens = [t[2] for t in test_loader.dataset.origin_tokens[idx]]
        f_i2s.write('Query: {}, {}\n'.format(doc_id, ' '.join(tokens)))
        f_s2i.write('Query: {}, {}\n'.format(doc_id, ' '.join(tokens)))
        for i2s_idx in i2s_idxs:
          doc_id = test_loader.dataset.doc_ids[i2s_idx]
          tokens = [t[2] for t in test_loader.dataset.origin_tokens[i2s_idx]]
          f_i2s.write('Key: {}, {}\n'.format(doc_id, ' '.join(tokens)))
        f_i2s.write('\n')

        for s2i_idx in s2i_idxs:
          doc_id = test_loader.dataset.doc_ids[s2i_idx]
          tokens = [t[2] for t in test_loader.dataset.origin_tokens[s2i_idx]]
          f_s2i.write('Key: {}, {}\n'.format(doc_id, ' '.join(tokens)))
        f_s2i.write('\n')

    I2S_eval = RetrievalEvaluation(I2S_idxs)
    S2I_eval = RetrievalEvaluation(S2I_idxs)
    I2S_r1 = I2S_eval.get_recall_at_k(1) 
    I2S_r5 = I2S_eval.get_recall_at_k(5)
    I2S_r10 = I2S_eval.get_recall_at_k(10)
    I2S_r30 = 0 # I2S_eval.get_recall_at_k(30)
    I2S_r50 = 0 # I2S_eval.get_recall_at_k(50)

    S2I_r1 = S2I_eval.get_recall_at_k(1) 
    S2I_r5 = S2I_eval.get_recall_at_k(5)
    S2I_r10 = S2I_eval.get_recall_at_k(10)
    S2I_r30 = 0 # S2I_eval.get_recall_at_k(30)
    S2I_r50 = 0 # S2I_eval.get_recall_at_k(50)

    print('Number of article-video pairs: {}'.format(I2S_idxs.size(0)))
    print('I2S recall@1={}\tI2S recall@5={}\tI2S recall@10={}\tI2S recall@30={}\tI2S recall@50={}'.format(I2S_r1, I2S_r5, I2S_r10, I2S_r30, I2S_r50)) 
    print('S2I recall@1={}\tS2I recall@5={}\tS2I recall@10={}\tS2I recall@30={}\tS2I recall@50={}'.format(S2I_r1, S2I_r5, S2I_r10, S2I_r30, S2I_r50)) 
    logger.info('Number of article-video pairs: {}'.format(I2S_idxs.size(0)))
    logger.info('I2S recall@1={}\tI2S recall@5={}\tI2S recall@10={}\tI2S recall@30={}\tI2S recall@50={}'.format(I2S_r1, I2S_r5, I2S_r10, I2S_r30, I2S_r50)) 
    logger.info('S2I recall@1={}\tS2I recall@5={}\tS2I recall@10={}\tS2I recall@30={}\tS2I recall@50={}'.format(S2I_r1, S2I_r5, S2I_r10, S2I_r30, S2I_r50)) 
  return I2S_r10, S2I_r10


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
    os.makedirs(config['model_path'])
  if not os.path.isdir(os.path.join(config['model_path'], 'log')):
    os.makedirs(os.path.join(config['model_path'], 'log')) 
  
  pred_out_dir = os.path.join(config['model_path'], 'pred_conll')
  if not os.path.isdir(pred_out_dir):
    os.makedirs(pred_out_dir)

  logging.basicConfig(filename=os.path.join(config['model_path'],'log/{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                      format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) 

  # Initialize dataloaders
  train_set = StarFeatureDataset(os.path.join(config['data_folder'], 'train.json'), 
                                                os.path.join(config['data_folder'], 'train_mixed.json'), 
                                                os.path.join(config['data_folder'], 'train_bboxes.json'),
                                                config, split='train')
  test_set = StarFeatureDataset(os.path.join(config['data_folder'], 'test.json'), 
                                               os.path.join(config['data_folder'], 'test_mixed.json'), 
                                               os.path.join(config['data_folder'], 'test_bboxes.json'), 
                                               config, split='test')
 
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

  # Initialize models
  text_model = nn.TransformerEncoderLayer(d_model=config.hidden_layer, nhead=1)
  image_model = VisualEncoder(400, config.hidden_layer)

  mention_model = SpanEmbedder(config, device)
  coref_model = StarTransformerClassifier(config).to(device)

  if config['training_method'] in ('pipeline', 'continue'):
      # text_model.load_state_dict(torch.load(config['text_model_path'], map_location=device))
      # for p in text_model.parameters():
      #   p.requires_grad = False
      # image_model.load_state_dict(torch.load(config['image_model_path'], map_location=device))
      mention_model.load_state_dict(torch.load(config['mention_model_path']))
      for p in mention_model.parameters():
        p.requires_grad = False
      coref_model.load_state_dict(torch.load(config['coref_model_path'], map_location=device))
      for p in coref_model.parameters():
        p.requires_grad = False

  # Training
  train(text_model, mention_model, image_model, coref_model, train_loader, test_loader, args)
