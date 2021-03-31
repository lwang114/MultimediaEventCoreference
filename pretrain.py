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
from image_models import VisualEncoder
from criterion import GraphCPCLoss 
from corpus_text import TextFeatureDataset 
from corpus_graph import StarFeatureDataset
from evaluator import Evaluation, RetrievalEvaluation, CoNLLEvaluation
from utils import make_prediction_readable

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

def train(text_model, mention_model, image_model, train_loader, test_loader, args, random_seed=None):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  if random_seed:
      config.random_seed = random_seed
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(True)
  
  if not isinstance(text_model, torch.nn.DataParallel):
    text_model = nn.DataParallel(text_model)

  if not isinstance(mention_model, torch.nn.DataParallel):
    mention_model = nn.DataParallel(mention_model)

  if not isinstance(image_model, torch.nn.DataParallel):
    image_model = nn.DataParallel(image_model)
  
  text_model.to(device)
  mention_model.to(device)
  image_model.to(device)

  # Define the training criterion
  token_dim = config.bert_hidden_size
  mention_dim = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2
  if config.with_mention_width:
    mention_layer += config.embedding_dimension
  if config.get('with_type_embedding', False):
    mention_layer += config.type_embedding_dimension
  event_event_criterion = CPCLoss(nPredicts=config.num_predict_event,
                                  dimOutputAR=mention_dim,
                                  dimOutputEncoder=mention_dim,
                                  negativeSamplingExt=config.num_negative_samples) # TODO
  event_argument_criterion = CPCLoss(nPredicts=1,
                                     dimOutputAR=mention_dim,
                                     dimOutputEncoder=mention_dim,
                                     negativeSamplingExt=config.num_negative_samples,
                                     auxiliaryEmbedding=config.role_embedding_dimension, # TODO
                                     nAuxiliary=len(train_loader.dataset.role_to_idx),
                                     startOffset=0)

  # entity_criterion = CPCLoss(dimOutputAR=mention_dim,
  #                            dimOutputEncoder=token_dim,
  #                            config) # TODO
  multimedia_criterion = TripletLoss(config)

  # Set up the optimizer
  if config.multimedia:
    optimizer = get_optimizer(config, [text_model, image_model, mention_model, coref_model])
  else:
    optimizer = get_optimizer(config, [text_model, mention_model, coref_model])
  
  # Start training
  total_loss = 0.
  total = 0.
  best_text_f1 = 0.
  best_grounding_f1 = 0.
  best_retrieval_recall = 0.
  results = {}
  begin_time = time.time()
  if args.evaluate_only:
    config.epochs = 0
  for epoch in range(args.start_epoch, config.epochs):
    text_model.train()
    mention_model.train()
    image_model.train()
    coref_model.train()
    for i, batch in enumerate(train_loader):
      if config.multimedia:
        doc_embeddings,\
        start_mappings,\
        end_mappings,\
        continuous_mappings,\
        event_mappings,\
        entity_mappings,\
        event_to_roles_mappings,\
        width, videos,\
        text_labels,\
        type_labels,\ 
        role_labels,\
        img_labels,\
        text_mask,\
        span_mask,\
        video_mask = batch # TODO Add entity attributes 
  
        videos = videos.to(device)
        img_labels = img_labels.to(device)
        video_mask = video_mask.to(device)
        first_grounding_idx,\
        second_grounding_idx,\
        pairwise_grounding_labels = get_pairwise_labels(text_labels, img_labels, is_training=False, device=device)
        pairwise_grounding_labels = pairwise_grounding_labels.to(torch.float)
        if first_grounding_idx is None:
          continue
        video_output = image_model(videos)
      else:
        doc_embeddings,\
        start_mappings,\
        end_mappings,\
        continuous_mappings,\
        event_mappings,\
        entity_mappings,\
        event_to_roles_mappings,\
        width, text_labels,\
        type_labels,\
        role_labels,\
        text_mask,\
        span_mask = batch # TODO

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
      role_labels = role_labels.to(device)
      text_mask = text_mask.to(device)
      span_mask = span_mask.to(device)
      first_text_idx, second_text_idx, pairwise_text_labels = get_pairwise_text_labels(text_labels, is_training=False, device=device) 
      pairwise_text_labels = pairwise_text_labels.to(torch.float).flatten()
      if first_text_idx is None:
        continue
      optimizer.zero_grad()

      text_output = text_model(doc_embeddings)
      mention_output = mention_model(doc_embeddings,
                                     start_mappings, end_mappings,
                                     continuous_mappings, width,
                                     type_labels=type_labels)
      context_output = mention_model(text_output,
                                     start_mappings, end_mappings,
                                     continuous_mappings, width,
                                     type_labels=type_labels)

      context_event_output = torch.matmul(event_mappings, context_output).view(batch_size*max_event_num, -1)

      ## Event-event prediction
      batch_size, max_event_num, _ = context_event_output.size() 
      event_mask = torch.matmul(event_mappings, span_mask)
      # (batch size, max num. of events, embed dim)
      event_output = torch.matmul(event_mappings, mention_output)
      ee_loss, _ = event_event_criterion(context_event_output.view(batch_size*max_event_num, -1),
                                         event_output.view(batch_size*max_event_num, -1),
                                         event_mask.view(batch_size*max_event_num, 1))

      ## Event-argument prediction
      max_role_num = event_to_roles_mappings.size(2)
      role_mask = torch.matmul(event_to_roles_mappings, span_mask.unsqueeze(1))
      # (batch size, max num. of events, max num. of roles, embed dim)
      role_output = torch.matmul(event_to_role_mappings, mention_output.unsqueeze(1))
      ea_loss, _ = event_argument_criterion(context_event_output.view(batch_size*max_event_num,-1).\
      unsqueeze(1).expand(1, max_role_num, 1),
                                            role_output.view(batch_size*max_event_num, -1),
                                            role_mask.view(batch_size*max_event_num, -1),
                                            role_labels)
      loss = ee_loss + ea_loss

      if self.multimedia:
        loss = loss + multimedia_criterion(doc_embeddings, video_output, text_mask, video_mask) 
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

    torch.save(text_model.module.state_dict(), '{}/text_model-{}.pth'.format(config['model_path'], config['random_seed']))
    torch.save(mention_model.module.state_dict(), '{}/mention_model-{}.pth'.format(config['model_path'], config['random_seed']))
    if config.multimedia:
      torch.save(image_model.module.state_dict(), '{}/image_model-{}.pth'.format(config['model_path'], config['random_seed']))

 
    if epoch % 1 == 0:
      task = config.get('task', 'coreference')
      if task in ('coreference', 'both'):
        res = test(text_model, mention_model, image_model, 
                   event_event_criterion, event_argument_criterion,
                   test_loader, args)
        if res['pairwise'][-1] >= best_text_f1:
          best_text_f1 = res['pairwise'][-1]
          results['pairwise'] = res['pairwise']
          results['muc'] = res['muc']
          results['ceafe'] = res['ceafe']
          results['bcubed'] = res['bcubed']
          results['avg'] = res['avg']
          torch.save(text_model.module.state_dict(), '{}/best_text_model-{}.pth'.format(config['model_path'], config['random_seed']))
          torch.save(mention_model.module.state_dict(), '{}/best_mention_model-{}.pth'.format(config['model_path'], config['random_seed']))
          if config.multimedia:
            torch.save(image_model.module.state_dict(), '{}/best_image_model-{}.pth'.format(config['model_path'], config['random_seed']))
        # TODO Print the negative sample classification result
        print('Best text coreference F1={}'.format(best_text_f1))

  if args.evaluate_only:
    results = test(text_model, mention_model, image_model,\
                   event_event_criterion, event_argument_criterion,
                   test_loader, args)
  return results
      
def test(text_model, mention_model, image_model, 
         event_event_criterion, event_argument_criterion,
         test_loader, args): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = pyhocon.ConfigFactory.parse_file(args.config)
    documents = test_loader.dataset.documents
    all_scores = []
    all_labels = []

    text_model.eval()
    image_model.eval()

    conll_eval_event = CoNLLEvaluation()
    f_out = open(os.path.join(config['model_path'], 'prediction.readable'), 'w')
    best_f1 = 0.
    results = {} 
    with torch.no_grad():     
        for i, batch in enumerate(test_loader):
          if config.multimedia:
            doc_embeddings,\
            start_mappings,\
            end_mappings,\
            continuous_mappings,\
            event_mappings,\
            entity_mappings,\
            event_to_roles_mappings,\
            width, videos,\
            text_labels,\
            type_labels,\ 
            role_labels,\
            img_labels,\
            text_mask,\
            span_mask,\
            video_mask = batch # TODO   

            videos = videos.to(device)
            img_labels = img_labels.to(device)
            video_mask = video_mask.to(device)
            
            first_grounding_idx, second_grounding_idx, pairwise_grounding_labels = get_pairwise_labels(text_labels, img_labels, is_training=False, device=device)
            pairwise_grounding_labels = pairwise_grounding_labels.to(torch.float)
            if first_grounding_idx is None:
              continue
            video_output = image_model(videos)
          else:
            doc_embeddings,\
            start_mappings,\
            end_mappings,\
            continuous_mappings,\
            event_mappings,\
            entity_mappings,\
            event_to_roles_mappings,\
            width, text_labels,\
            type_labels,\
            role_labels,\
            text_mask,\
            span_mask = batch # TODO

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
          role_labels = role_labels.to(device)
          text_mask = text_mask.to(device)
          span_mask = span_mask.to(device)

           # Extract span and video embeddings
          text_output = text_model(doc_embeddings) 
          mention_output = mention_model(doc_embeddings,
                                         start_mappings, end_mappings,
                                         continuous_mappings, width,
                                         type_labels=type_labels)
          context_output = mention_model(text_output,
                                         start_mappings, end_mappings,
                                         continuous_mappings, width,
                                         type_labels=type_labels)
         
          event_labels = torch.bmm(event_mappings, text_labels.unsqueeze(-1).float()).squeeze(-1).long()
          event_num = event_mappings.sum(-1).sum(-1).long()

          B = doc_embeddings.size(0)     
          for idx in range(B):
            global_idx = i * test_loader.batch_size + idx
            
            # Compute pairwise labels 
            first_event_idx, second_event_idx, pairwise_event_labels = get_pairwise_text_labels(event_labels[idx, :event_num[idx]].unsqueeze(0), 
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
              role_label = role_labels[idx, :mention_num]
 
              # Compute pairwise scores
              event_antecedents, event_scores = coref_model.module.predict_cluster(context_output[idx],
                                                                                   mention_output[idx],
                                                                                   first_event_idx, 
                                                                                   second_event_idx) 
              event_antecedents = torch.LongTensor(event_antecedents)
              all_event_scores.append(event_scores)
              all_event_labels.append(pairwise_event_labels.to(torch.int).cpu())
              pred_event_clusters, gold_event_clusters = conll_eval_event(event_start_ends,
              event_antecedents,
              event_start_ends,
              event_label)
            else:
              pred_event_clusters, gold_event_clusters = [], []

          all_event_scores = torch.cat(all_event_scores)
          all_event_labels = torch.cat(all_event_labels)

          strict_preds_event = (all_event_scores > 0).to(torch.int)
          eval_event = Evaluation(strict_preds_event, all_event_labels.to(device))
          print('Number of predictions: {}/{}'.format(strict_preds_event.sum(), len(strict_preds_event)))
          print('Number of positive pairs: {}/{}'.format(len((all_event_labels == 1).nonzero()),
                                                     len(all_event_labels)))
        print('Pairwise - Recall: {}, Precision: {}, F1: {}'.format(eval_event.get_recall(),
                                                                eval_event.get_precision(), eval_event.get_f1()))
        
        muc, b_cubed, ceafe, avg = conll_eval_event.get_metrics()
        results['pairwise'] = (eval_event.get_precision().item(), eval_event.get_recall().item(), eval_event.get_f1().item())
        results['muc'] = muc
        results['bcubed'] = b_cubed
        results['ceafe'] = ceafe
        results['avg'] = avg
        conll_metrics = muc+b_cubed+ceafe+avg
        print('MUC - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'Bcubed - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CEAFe - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '
              'CoNLL - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(*conll_metrics)) 
        return results

if __name__ == '__main__':
