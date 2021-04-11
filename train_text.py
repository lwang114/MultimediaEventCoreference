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
from text_models import NoOp, SpanEmbedder, StarSimplePairWiseClassifier, SimplePairWiseClassifier, StarTransformerClassifier
from corpus_text import TextFeatureDataset
from evaluator import Evaluation, RetrievalEvaluation, CoNLLEvaluation
from conll import write_output_file
from copy import deepcopy
from utils import create_type_to_idx, create_role_to_idx

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
  # image_model.to(device)
  coref_model.to(device)

  # Create/load exp
  if not os.path.isdir(args.exp_dir):
    os.path.mkdir(args.exp_dir)

  if args.start_epoch != 0:
    text_model.load_state_dict(torch.load('{}/text_model_best.pth'.format(args.exp_dir, args.start_epoch)))

  # Define the training criterion
  criterion = nn.BCEWithLogitsLoss()
   
  # Start training
  total_loss = 0.
  total = 0.
  best_f1 = 0.
  begin_time = time.time()
  if args.evaluate_only:
      config.epochs = 0
      optimizer = None
  else:
      # Set up the optimizer  
      optimizer = get_optimizer(config, [text_model, mention_model, coref_model])
    
  best_f1 = 0
  for epoch in range(args.start_epoch, config.epochs):
    text_model.train()
    image_model.train()
    coref_model.train()
    for i, batch in enumerate(train_loader):
      doc_embeddings,\
      start_mappings, end_mappings,\
      continuous_mappings,\
      event_mappings, entity_mappings,\
      event_to_roles_mappings,\
      width,\
      text_labels, type_labels,\
      role_labels,\
      text_mask, span_mask = batch 

      B = doc_embeddings.size(0)
      doc_embeddings = doc_embeddings.to(device)
      start_mappings = start_mappings.to(device)
      end_mappings = end_mappings.to(device)
      continuous_mappings = continuous_mappings.to(device)
      event_mappings = event_mappings.to(device)
      entity_mappings = entity_mappings.to(device)
      event_to_roles_mappings = event_to_roles_mappings.to(device)
      width = width.to(device)
      text_labels = text_labels.to(device)
      type_labels = type_labels.to(device)
      span_mask = span_mask.to(device)
      span_num = span_mask.sum(-1).long()

      event_labels = torch.bmm(event_mappings, text_labels.unsqueeze(-1).float()).squeeze(-1).long()
      entity_labels = torch.bmm(entity_mappings, text_labels.unsqueeze(-1).float()).squeeze(-1).long()
      first_event_idx, second_event_idx, pairwise_event_labels = get_pairwise_labels(event_labels, is_training=False, device=device)
      first_entity_idx, second_entity_idx, pairwise_entity_labels = get_pairwise_labels(entity_labels, is_training=False, device=device)      
      
      pairwise_event_labels = pairwise_event_labels.to(torch.float).flatten()
      pairwise_entity_labels = pairwise_entity_labels.to(torch.float).flatten()
      
      optimizer.zero_grad()

      text_output = text_model(doc_embeddings)
      mention_output = mention_model(text_output,
                                     start_mappings, end_mappings,
                                     continuous_mappings, width,
                                     type_labels=type_labels)
          
      if not first_event_idx is None:
          if config.classifier.split('_')[0] == 'graph':
            event_scores = coref_model(mention_output,
                                       event_mappings,
                                       event_to_roles_mappings,
                                       first_event_idx[0],
                                       second_event_idx[0]).flatten()
          else:
            event_scores = []
            for idx in range(B):
              event_scores.append(coref_model(mention_output[idx, first_event_idx[idx]],
                                              mention_output[idx, second_event_idx[idx]]))
            event_scores = torch.cat(event_scores).squeeze(1)
          loss_event = criterion(event_scores, pairwise_event_labels)
      else:
          loss_event = torch.zeros((mention_output.size(0),), dtype=torch.float, device=device)
          
      if not first_entity_idx is None:
          if config.classifier.split('_')[0] == 'graph':
            entity_scores = coref_model(mention_output,
                                        entity_mappings,
                                        entity_mappings.unsqueeze(-2),
                                        first_entity_idx[0],
                                        second_entity_idx[0]).flatten()
          else:
            entity_scores = []
            for idx in range(B):
              entity_scores.append(coref_model(mention_output[idx, first_entity_idx[idx]],
                                               mention_output[idx, second_entity_idx[idx]]))
            entity_scores = torch.cat(entity_scores).squeeze(1)
          loss_entity = criterion(entity_scores, pairwise_entity_labels)
      else:
          loss_entity = torch.zeros((mention_output.size(0),), dtype=torch.float, device=device)
      loss = loss_event + loss_entity
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
    torch.save(coref_model.module.state_dict(), '{}/coref_model.pth'.format(args.exp_dir))
    if epoch % 1 == 0:
      pairwise_f1 = test(text_model, mention_model, image_model, coref_model, test_loader, args)
      if pairwise_f1 > best_f1:
          best_f1 = pairwise_f1
          torch.save(text_model.module.state_dict(), '{}/text_model_best.pth'.format(args.exp_dir, epoch))
          torch.save(image_model.module.state_dict(), '{}/image_model_best.pth'.format(args.exp_dir, epoch))
          torch.save(mention_model.module.state_dict(), '{}/mention_model_best.pth'.format(args.exp_dir, epoch))
          torch.save(coref_model.module.state_dict(), '{}/coref_model_best.pth'.format(args.exp_dir, epoch))
      print('Best F1: {}'.format(best_f1))
          
  if args.evaluate_only:
    pairwise_f1 = test(text_model, mention_model, image_model, coref_model, test_loader, args)

    
    
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
    f_out = open(os.path.join(args.exp_dir, 'prediction.readable'), 'w')
    with torch.no_grad(): 
      pred_dicts = []
      for i, batch in enumerate(test_loader):
        doc_embeddings,\
        start_mappings, end_mappings,\
        continuous_mappings,\
        event_mappings, entity_mappings,\
        event_to_roles_mappings, width,\
        text_labels, type_labels,\
        role_labels,\
        text_mask, span_mask = batch
        
        token_num = text_mask.sum(-1).long()
        span_num = span_mask.sum(-1).long()
        doc_embeddings = doc_embeddings.to(device)
        start_mappings = start_mappings.to(device)
        end_mappings = end_mappings.to(device)
        continuous_mappings = continuous_mappings.to(device)
        event_mappings = event_mappings.to(device)
        entity_mappings = entity_mappings.to(device)
        event_to_roles_mappings = event_to_roles_mappings.to(device)
        width = width.to(device)

        text_labels = text_labels.to(device)
        type_labels = type_labels.to(device)
        span_mask = span_mask.to(device)
        span_num = span_mask.sum(-1).long()

        event_labels = torch.bmm(event_mappings, text_labels.unsqueeze(-1).float()).squeeze(-1).long()
        entity_labels = torch.bmm(entity_mappings, text_labels.unsqueeze(-1).float()).squeeze(-1).long()
        event_num = event_mappings.sum(-1).sum(-1).long()
        entity_num = entity_mappings.sum(-1).sum(-1).long()
        
        # Compute mention embeddings
        text_output = text_model(doc_embeddings)
        mention_output = mention_model(text_output,
                                       start_mappings, end_mappings,
                                       continuous_mappings, width,
                                       type_labels)
            
        B = doc_embeddings.size(0)  
        for idx in range(B):
            global_idx = i * test_loader.batch_size + idx
            
            # Compute pairwise labels
            first_event_idx, second_event_idx, pairwise_event_labels = get_pairwise_labels(event_labels[idx, :event_num[idx]].unsqueeze(0), 
                                                                                           is_training=False, device=device)
            first_entity_idx, second_entity_idx, pairwise_entity_labels = get_pairwise_labels(entity_labels[idx, :entity_num[idx]].unsqueeze(0),
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
                if config.classifier.split('_')[0] == 'graph':
                  event_antecedents, event_scores = coref_model.module.predict_cluster(mention_output[idx],
                                                                                       event_mapping, event_to_roles_mapping,
                                                                                       first_event_idx, second_event_idx)
                else:
                  event_output = torch.mm(event_mapping, mention_output[idx])
                  event_antecedents, event_scores = coref_model.module.predict_cluster(event_output,
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
                if config.classifier.split('_')[0] == 'graph':
                  entity_antecedents, entity_scores = coref_model.module.predict_cluster(mention_output[idx],
                                                                                         entity_mapping, entity_mapping.unsqueeze(1),
                                                                                         first_entity_idx, second_entity_idx)
                else:
                  entity_output = torch.mm(entity_mapping, mention_output[idx])
                  entity_antecedents, entity_scores = coref_model.module.predict_cluster(entity_output,
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
                                                                                   tokens, arguments=test_loader.dataset.event_to_roles[doc_id])
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

    if config.classifier.split('_')[0] == 'graph': # TODO
      align(text_model, mention_model, image_model, coref_model, test_loader, args) 
    return eval_event.get_f1()

  def align(text_model, mention_model, image_model, coref_model, test_loader, args): # TODO
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = pyhocon.ConfigFactory.parse_file(args.config)
    documents = test_loader.dataset.documents
    pred_dicts = []
    text_model.eval()
    image_model.eval()
    coref_model.eval()

    with torch.no_grad():
      for i, batch in enumerate(test_loader):
        doc_embeddings,\
        start_mappings, end_mappings,\
        continuous_mappings,\
        event_mappings, entity_mappings,\
        event_to_roles_mappings, width,\
        text_labels, type_labels,\
        role_labels,\
        text_mask, span_mask = batch
        
        token_num = text_mask.sum(-1).long()
        span_num = span_mask.sum(-1).long()
        doc_embeddings = doc_embeddings.to(device)
        start_mappings = start_mappings.to(device)
        end_mappings = end_mappings.to(device)
        continuous_mappings = continuous_mappings.to(device)
        event_mappings = event_mappings.to(device)
        entity_mappings = entity_mappings.to(device)
        event_to_roles_mappings = event_to_roles_mappings.to(device)
        width = width.to(device)

        text_labels = text_labels.to(device)
        type_labels = type_labels.to(device)
        span_mask = span_mask.to(device)
        span_num = span_mask.sum(-1).long()

        event_labels = torch.bmm(event_mappings, text_labels.unsqueeze(-1).float()).squeeze(-1).long()
        event_num = event_mappings.sum(-1).sum(-1).long()
        
        text_output = text_model(doc_embeddings)
        mention_output = mention_model(text_output,
                                       start_mappings, end_mappings,
                                       continuous_mappings, width,
                                       type_labels)
        _, score_mats = coref_model(mention_output,
                                    event_mappings,
                                    event_to_roles_mappings,
                                    first_event_idx[0],
                                    second_event_idx[0],
                                    return_score_matrix=True)
            
        B = doc_embeddings.size(0)  
        for idx in range(B):
            global_idx = i * test_loader.batch_size + idx
            doc_id = test_loader.dataset.doc_ids[global_idx]
            tokens = [token[2] for token in test_loader.dataset.documents[doc_id]]
            
            first_event_idx, second_event_idx, pairwise_event_labels = get_pairwise_labels(event_labels[idx, :event_num[idx]].unsqueeze(0), 
                                                                                           is_training=False, device=device)
            if first_event_idx is None:
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
              
              # Compute alignment scores
              for pair_idx, (idx1, idx2, y) in enumerate(zip(first_event_idx, second_event_idx, pairwise_event_labels)):
                if y > 0:
                  n_args_first = event_to_roles_mapping[idx1].sum().long()
                  n_args_second = event_to_roles_mapping[idx2].sum().long()                 
                  if n_args_first <= 0 or n_args_second <= 0:
                    continue
                  first_arg_idxs = torch.max(event_to_roles_mapping[idx1], dim=-1)[1][:n_args_first]
                  first_arg_spans = [origin_candidate_start_ends[i] for i in first_arg_idxs]
                  first_arguments = [' '.join(tokens[s[0]:s[1]+1]) for s in first_arg_spans]


                  second_arg_idxs = torch.max(event_to_roles_mapping[idx2], dim=-1)[1][:n_args_second]
                  second_arg_spans = [origin_candidate_start_ends[i] for i in second_arg_idxs]
                  second_arguments = [' '.join(tokens[s[0]:s[1]+1]) for s in second_arg_spans]

                  first_trigger = ' '.join(tokens[origin_candidate_start_ends[idx1, 0]:origin_candidate_start_ends[idx1, 1]+1])
                  second_trigger = ' '.join(tokens[origin_candidate_start_ends[idx2, 0]:origin_candidate_start_ends[idx2, 1]+1])
                  score_mat = score_mats[idx, pair_idx, :n_args_first, :n_args_second].cpu().detach().numpy()
                  pred_dicts.append({'doc_id': doc_id,
                                     'score_mat': score_mat.tolist(),
                                     'first_trigger': first_trigger,
                                     'second_trigger': second_trigger,
                                     'first_arguments': first_arguments,
                                     'second_arguments': second_arguments})
    json.dump(pred_dicts, open(os.path.join(config['model_path'], 'graph_alignment_scores.json'), 'w'), indent=2)
      
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
  fix_seed(config)

  if not args.exp_dir:
    args.exp_dir = config['model_path']
  else:
    config['model_path'] = args.exp_dir
    
  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  if not os.path.isdir(config['log_path']):
    os.makedirs(config['log_path']) 
  
  pred_out_dir = os.path.join(config['model_path'], 'pred_conll')
  if not os.path.isdir(pred_out_dir):
    os.mkdir(pred_out_dir)

  logging.basicConfig(filename=os.path.join(config['log_path'],'{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                      format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) 

  # Initialize dataloaders
  splits = [os.path.join(config['data_folder'], 'train_mixed.json'),\
            os.path.join(config['data_folder'], 'test_mixed.json')]
  type_to_idx = create_type_to_idx(splits) 
  role_to_idx = create_role_to_idx(splits)
  train_set = TextFeatureDataset(os.path.join(config['data_folder'], 'train.json'), splits[0],
                                 config, split='train', type_to_idx=type_to_idx, role_to_idx=role_to_idx)
  config_test = deepcopy(config)
  config_test['is_one_indexed'] = True if config['data_folder'].split('/')[-2] == 'ecb' else False
  print(config_test['is_one_indexed'])
  test_set = TextFeatureDataset(os.path.join(config['data_folder'], 'test.json'), splits[1],
                                config_test, split='test', type_to_idx=type_to_idx, role_to_idx=role_to_idx)

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=False)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=False)

  # Initialize models
  embedding_dim = config.hidden_layer
  text_model = nn.TransformerEncoderLayer(d_model=config.hidden_layer, nhead=1)
  mention_model = SpanEmbedder(config, device).to(device)

  image_model = NoOp()
  classifier = config.get('classifier', 'graph')
  if classifier == 'simple':
    coref_model = SimplePairWiseClassifier(config).to(device)
  elif classifier == 'graph':
    coref_model = StarSimplePairWiseClassifier(config).to(device)
  elif classifier == 'graph_transformer':
    coref_model = StarTransformerClassifier(config).to(device)

  if config['training_method'] in ('pipeline', 'continue'):
      text_model.load_state_dict(torch.load(config['text_model_path'], map_location=device))
      for p in text_model.parameters():
        p.requires_grad = False
      mention_model.load_state_dict(torch.load(config['mention_model_path']))  
      for p in mention_model.parameters():
        p.requires_grad = False
      coref_model.load_state_dict(torch.load(config['coref_model_path'], map_location=device))
      for p in coref_model.parameters():
        p.requires_grad = False
      
  # Training
  n_params = 0
  for p in text_model.parameters():
      n_params += p.numel()
  
  for p in mention_model.parameters():
      n_params += p.numel()

  for p in coref_model.parameters():
      n_params += p.numel()

  print('Number of parameters in coref classifier: {}'.format(n_params))
  train(text_model, mention_model, image_model, coref_model, train_loader, test_loader, args)
