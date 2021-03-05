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
from models import SpanEmbedder, MultimediaPairWiseClassifier
from pairwise_grounder import PairwiseGrounder
from text_models import BiLSTM, NoOp
from corpus_supervised import SupervisedGroundingFeatureDataset
from corpus_supervised_glove import SupervisedGroundingGloveFeatureDataset
from evaluator import Evaluation, RetrievalEvaluation
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

def get_pairwise_text_labels(labels_batch, is_training, device):
  B = labels_batch.size(0)
  first_batch = []
  second_batch = []
  pairwise_labels_batch = []
  for idx in range(B):
    labels = labels_batch[idx]
    first, second = zip(*list(combinations(range(len(labels)), 2))) 
    first = torch.tensor(first)
    second = torch.tensor(second)
    pairwise_labels = (labels[first] != 0) & (labels[second] != 0) & \
                      (labels[first] == labels[second])
    pairwise_labels = pairwise_labels.to(torch.long).to(device)
    positives = (pairwise_labels == 1).nonzero().squeeze()

    if config['loss'] == 'hinge' and is_training and torch.sum(pairwise_labels) > 0: 
        positive_ratio = int(torch.sum(pairwise_labels).cpu().detach().numpy()) / len(first)
        negatives = (pairwise_labels != 1).nonzero()
        if len(negatives) == 0:
          continue
        negatives = negatives.squeeze()
        rands = torch.rand(len(negatives))
        rands = (rands < positive_ratio * 20).to(torch.long)
        sampled_negatives = negatives[rands.nonzero().squeeze()]
        pairwise_labels[sampled_negatives] = torch.tensor(-1, device=device)
    else:
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(0, device=device))
    torch.cuda.empty_cache()
    first_batch.append(first)
    second_batch.append(second)
    pairwise_labels_batch.append(pairwise_labels)

  first_batch = torch.stack(first_batch, dim=0)
  second_batch = torch.stack(second_batch, dim=0)
  pairwise_labels_batch = torch.stack(pairwise_labels_batch, dim=0)
  return first_batch, second_batch, pairwise_labels_batch


def train(text_model, mention_model, image_model, grounding_model, coref_model, train_loader, test_loader, args):
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
      doc_embeddings, start_mappings, end_mappings, continuous_mappings,\
      width, videos,\
      text_labels, img_labels,\
      text_mask, span_mask, video_mask = batch   

      B = doc_embeddings.size(0)     
      doc_embeddings = doc_embeddings.to(device)
      start_mappings = start_mappings.to(device)
      end_mappings = end_mappings.to(device)
      continuous_mappings = continuous_mappings.to(device)

      width = width.to(device)
      videos = videos.to(device)
      text_labels = text_labels.to(device)
      img_labels = img_labels.to(device)
      text_mask = text_mask.to(device)
      span_mask = span_mask.to(device)
      video_mask = video_mask.to(device)
      first_grounding_idx, second_grounding_idx, pairwise_grounding_labels = get_pairwise_labels(text_labels, img_labels, is_training=False, device=device)
      first_text_idx, second_text_idx, pairwise_text_labels = get_pairwise_text_labels(text_labels, is_training=True, device=device)      
      pairwise_grounding_labels = pairwise_grounding_labels.to(torch.float)
      pairwise_text_labels = pairwise_text_labels.to(torch.float).flatten()
      if first_grounding_idx is None or first_text_idx is None:
        continue

      optimizer.zero_grad()

      text_output = text_model(doc_embeddings)
      video_output = image_model(videos)
      mention_output = mention_model(text_output, start_mappings, end_mappings, continuous_mappings, width)
      mml_loss, text_output_i2s = grounding_model(text_output, video_output, 
                                                  text_mask, video_mask)
      mention_output_i2s = mention_model(text_output_i2s, start_mappings, end_mappings, continuous_mappings, width)
      
      first = torch.cat([mention_output[idx, first_text_idx[idx]] for idx in range(B)])
      second = torch.cat([mention_output[idx, second_text_idx[idx]] for idx in range(B)])
      first_i2s = torch.cat([mention_output_i2s[idx, first_text_idx[idx]] for idx in range(B)])
      second_i2s = torch.cat([mention_output_i2s[idx, second_text_idx[idx]] for idx in range(B)])
      text_scores = coref_model(first, second) + coref_model(first_i2s, second_i2s)
      text_scores = text_scores.squeeze(-1)

      # bce_grounding_loss = criterion(grounding_scores.view(B, -1), pairwise_grounding_labels)
      # bce_grounding_loss = bce_grounding_loss / (B * span_mask.size(1))
      bce_text_loss = criterion(text_scores, pairwise_text_labels)
      loss = mml_loss + bce_text_loss
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
    torch.save(mention_model.module.state_dict(), '{}/mention_model.{}.pth'.format(args.exp_dir, epoch))
    torch.save(coref_model.module.state_dict(), '{}/text_scorer.{}.pth'.format(args.exp_dir, epoch))
 
    if epoch % 5 == 0:
      task = config.get('task', 'coreference')
      if task in ('coreference', 'both'):
        text_f1 = test(text_model, mention_model, image_model, grounding_model, coref_model, test_loader, args)
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
      text_f1 = test(text_model, mention_model, image_model, grounding_model, coref_model, test_loader, args)
    if task in ('retrieval', 'both'):
      I2S_r10, S2I_r10 = test_retrieve(text_model, image_model, grounding_model, test_loader, args)


def test(text_model, mention_model, image_model, grounding_model, coref_model, test_loader, args): 
    config = pyhocon.ConfigFactory.parse_file(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    documents = test_loader.dataset.documents
    best_f1 = 0.

    text_model.eval()
    image_model.eval()
    coref_model.eval()
    
    out_dir_text_only = os.path.join(config['model_path'], 'pred_conll_text_only')
    if not os.path.isdir(out_dir_text_only):
      os.makedirs(out_dir_text_only)
        
    out_dir = os.path.join(config['model_path'], 'pred_conll')
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)

    with torch.no_grad():
        all_grounding_scores = []
        all_grounding_labels = []
        all_text_only_scores = []
        all_text_scores = []
        all_text_labels = []
        pred_text_dicts = []
        pred_text_only_dicts = []
        for i, batch in enumerate(test_loader):
          doc_embeddings,\
          start_mappings,\
          end_mappings,\
          continuous_mappings,\
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
          text_mask = text_mask.to(device)
          span_mask = span_mask.to(device)
          video_mask = video_mask.to(device)

          # Extract span and video embeddings
          text_output = text_model(doc_embeddings)
          video_output = image_model(videos)
          mention_output = mention_model(text_output, start_mappings, end_mappings, continuous_mappings, width)
          _, text_output_i2s = grounding_model(text_output, 
                                               video_output,
                                               text_mask,
                                               video_mask)
          mention_output_i2s = mention_model(text_output_i2s, start_mappings, end_mappings, continuous_mappings, width)

          # Compute score for each span pair
          B = doc_embeddings.size(0) 
          for idx in range(B):
            if span_num[idx] <= 1:
              print('No coref labels for doc {} with {} spans'.format(idx, span_num[idx]))
              continue
            
            first_text_idx, second_text_idx, pairwise_text_labels = get_pairwise_text_labels(text_labels[idx, :span_num[idx]].unsqueeze(0), 
                                                                                             is_training=False, device=device)
            first_text_idx = first_text_idx.squeeze(0)
            second_text_idx = second_text_idx.squeeze(0)
            pairwise_text_labels = pairwise_text_labels.squeeze(0)
            clusters, text_scores, text_only_scores = coref_model.module.predict_cluster( 
                                                       mention_output[idx, :span_num[idx]],
                                                       mention_output_i2s[idx, :span_num[idx]],
                                                       first_text_idx,
                                                       second_text_idx)
            clusters_text_only, _, _ = coref_model.module.predict_cluster( 
                                                       mention_output[idx, :span_num[idx]],
                                                       mention_output_i2s[idx, :span_num[idx]],
                                                       first_text_idx,
                                                       second_text_idx, w=0)

            all_text_only_scores.append(text_only_scores.cpu())
            all_text_scores.append(text_scores.cpu())
            all_text_labels.append(pairwise_text_labels.to(torch.int).cpu())
            
            global_idx = i * test_loader.batch_size + idx
            doc_id = test_loader.dataset.doc_ids[global_idx]
            origin_tokens = [token[2] for token in test_loader.dataset.origin_tokens[global_idx]]
            candidate_start_ends = test_loader.dataset.origin_candidate_start_ends[global_idx]
            doc_name = doc_id
            document = {doc_id:test_loader.dataset.documents[doc_id]}
      
            write_output_file(document, clusters,
                              [doc_id]*candidate_start_ends.shape[0],
                              candidate_start_ends[:, 0].tolist(),
                              candidate_start_ends[:, 1].tolist(),
                              out_dir,
                              doc_name, 
                              False, True) 
            write_output_file(document, clusters_text_only,
                              [doc_id]*candidate_start_ends.shape[0],
                              candidate_start_ends[:, 0].tolist(),
                              candidate_start_ends[:, 1].tolist(),
                              out_dir_text_only,
                              doc_name, 
                              False, True)

            pred_text_only_dicts.append({'doc_id': doc_id,
                               'first_idx': first_text_idx.cpu().detach().numpy().tolist(),
                               'second_idx': second_text_idx.cpu().detach().numpy().tolist(),
                               'tokens': origin_tokens,
                               'mention_spans': candidate_start_ends.tolist(),
                               'score': text_only_scores.flatten().cpu().detach().numpy().tolist(),
                               'pairwise_label': pairwise_text_labels.cpu().detach().numpy().tolist()})

            pred_text_dicts.append({'doc_id': doc_id,
                               'first_idx': first_text_idx.cpu().detach().numpy().tolist(),
                               'second_idx': second_text_idx.cpu().detach().numpy().tolist(),
                               'tokens': origin_tokens,
                               'mention_spans': candidate_start_ends.tolist(),
                               'score': text_scores.flatten().cpu().detach().numpy().tolist(),
                               'pairwise_label': pairwise_text_labels.cpu().detach().numpy().tolist()})

        all_text_scores = torch.cat(all_text_scores)
        all_text_only_scores = torch.cat(all_text_only_scores)
        all_text_labels = torch.cat(all_text_labels)
        
        strict_text_only_preds = (all_text_only_scores > 0).to(torch.int)
        eval = Evaluation(strict_text_only_preds, all_text_labels)
        print('Number of text-only predictions: {}/{}'.format(strict_text_only_preds.sum(), len(strict_text_only_preds)))
        print('Number of text positive pairs: {}/{}'.format(len((all_text_labels == 1).nonzero()),
                                                       len(all_text_labels)))
        print('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                  eval.get_precision(), eval.get_f1()))
        logger.info('Number of text-only predictions: {}/{}'.format(strict_text_only_preds.sum(), len(strict_text_only_preds)))
        logger.info('Number of text positive pairs: {}/{}'.format(len((all_text_labels == 1).nonzero()),
                                                       len(all_text_labels)))
        logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                  eval.get_precision(), eval.get_f1())) 
        strict_text_preds = (all_text_scores > 0).to(torch.int)
        eval = Evaluation(strict_text_preds, all_text_labels)
        print('Number of text predictions: {}/{}'.format(strict_text_preds.sum(), len(strict_text_preds)))
        print('Number of text positive pairs: {}/{}'.format(len((all_text_labels == 1).nonzero()),
                                                       len(all_text_labels)))
        print('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                  eval.get_precision(), eval.get_f1()))
        logger.info('Number of text predictions: {}/{}'.format(strict_text_preds.sum(), len(strict_text_preds)))
        logger.info('Number of text positive pairs: {}/{}'.format(len((all_text_labels == 1).nonzero()),
                                                             len(all_text_labels)))
        logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                        eval.get_precision(), eval.get_f1()))
        if eval.get_f1() > best_f1:
          out_file = os.path.join(args.exp_dir, '{}_prediction_text_only_coref.json'.format(args.config.split('.')[0].split('/')[-1]))
          json.dump(pred_text_only_dicts, open(out_file, 'w'), indent=4)

          out_file = os.path.join(args.exp_dir, '{}_prediction_text_coref.json'.format(args.config.split('.')[0].split('/')[-1]))
          json.dump(pred_text_only_dicts, open(out_file, 'w'), indent=4)
          best_f1 = eval.get_f1()
             
    return best_f1


def test_retrieve(text_model, image_model, grounding_model, test_loader, args):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  text_embeddings = []
  video_embeddings = []
  text_masks = []
  video_masks = []
  text_model.eval()
  image_model.eval()
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

    I2S_idxs, S2I_idxs = grounding_model.retrieve(text_embeddings, video_embeddings, text_masks, video_masks, k=10)
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
    os.mkdir(config['model_path'])
  if not os.path.isdir(os.path.join(config['model_path'], 'log')):
    os.mkdir(os.path.join(config['model_path'], 'log')) 
  
  pred_out_dir = os.path.join(config['model_path'], 'pred_conll')
  if not os.path.isdir(pred_out_dir):
    os.mkdir(pred_out_dir)

  logging.basicConfig(filename=os.path.join(config['model_path'],'log/{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                      format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) 

  # Initialize dataloaders
  type_to_idx = create_type_to_idx(os.path.join(config['data_folder'], 'train_mixed.json'))
  if config.get('glove_dimension', None):
      train_set = SupervisedGroundingGloveFeatureDataset(os.path.join(config['data_folder'], 'train.json'), 
                                                         os.path.join(config['data_folder'], 'train_mixed.json'), 
                                                         os.path.join(config['data_folder'], 'train_bboxes.json'), 
                                                         config, split='train')  

      test_set = SupervisedGroundingGloveFeatureDataset(os.path.join(config['data_folder'], 'test.json'), 
                                                        os.path.join(config['data_folder'], 'test_mixed.json'), 
                                                        os.path.join(config['data_folder'], 'test_bboxes.json'), 
                                                        config, split='test')
  else:
      train_set = SupervisedGroundingFeatureDataset(os.path.join(config['data_folder'], 'train.json'), 
                                                    os.path.join(config['data_folder'], 'train_mixed.json'), 
                                                    os.path.join(config['data_folder'], 'train_bboxes.json'), 
                                                    config, split='train', type_to_idx=type_to_idx)
      test_set = SupervisedGroundingFeatureDataset(os.path.join(config['data_folder'], 'test.json'), 
                                                   os.path.join(config['data_folder'], 'test_mixed.json'), 
                                                   os.path.join(config['data_folder'], 'test_bboxes.json'), 
                                                   config, split='test', type_to_idx=type_to_idx)
 
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

  # Initialize models
  embedding_dim = config.crossmedia_layer

  if config.get('glove_dimension', None):
    text_model = BiLSTM(config.glove_dimension, embedding_dim // 2) 
  else:
    text_model = BiLSTM(1024, embedding_dim // 2)

  if config['img_feat_type'] == 'resnet34':
    image_model = nn.Linear(512, embedding_dim)
  elif config['img_feat_type'] == 'mmaction_feat': 
    image_model = NoOp() # nn.Linear(400, 400)
  else:
    image_model = nn.Linear(2048, embedding_dim)
  mention_model = SpanEmbedder(config, device)
  grounding_model = PairwiseGrounder(config)
  coref_model = MultimediaPairWiseClassifier(config).to(device)

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
  train(text_model, mention_model, image_model, grounding_model, coref_model, train_loader, test_loader, args)