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
from joint_supervised_grounded_coreference import JointSupervisedGroundedCoreferencer, BiLSTM
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
    first_batch.append(first)
    second_batch.append(second)
    pairwise_labels_batch.append(pairwise_labels)

  first_batch = torch.stack(first_batch, dim=0)
  second_batch = torch.stack(second_batch, dim=0)
  pairwise_labels_batch = torch.stack(pairwise_labels_batch, dim=0)
  return first_batch, second_batch, pairwise_labels_batch


def train(text_model, mention_model, image_model, coref_model, train_loader, test_loader, args):
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
  criterion = nn.BCEWithLogitsLoss(reduction='sum')
  
  # Set up the optimizer  
  optimizer = get_optimizer(config, [text_model, image_model, coref_model])
   
  # Start training
  total_loss = 0.
  total = 0.
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

      optimizer.zero_grad()

      text_output = text_model(doc_embeddings)
      video_output = image_model(videos)

      start_embeddings = torch.matmul(start_mappings, text_output)
      end_embeddings = torch.matmul(end_mappings, text_output)
      start_end_embeddings = torch.cat([start_embeddings, end_embeddings], dim=-1)
      continuous_embeddings = torch.matmul(continuous_mappings, text_output.unsqueeze(1))
      mention_output = mention_model(start_end_embeddings, continuous_embeddings, width)

      mml_loss, grounding_scores, text_scores = coref_model(text_output, mention_output, video_output, 
                                                            text_mask, span_mask, video_mask)
      first_grounding_idx, second_grounding_idx, pairwise_grounding_labels = get_pairwise_labels(text_labels, img_labels, is_training=False, device=device)
      first_text_idx, second_text_idx, pairwise_text_labels = get_pairwise_text_labels(text_labels, is_training=False, device=device)

      if first_grounding_idx is None or first_text_idx is None:
        continue
      pairwise_grounding_labels = pairwise_grounding_labels.to(torch.float)
      pairwise_text_labels = pairwise_text_labels.to(torch.float)
      bce_grounding_loss = criterion(grounding_scores.view(B, -1), pairwise_grounding_labels)
      bce_grounding_loss = bce_grounding_loss / (B * span_mask.size(1))
      bce_text_loss = criterion(text_scores, pairwise_text_labels)
      loss = mml_loss # bce_grounding_loss
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
      task = config.get('task', 'coreference')
      if task == 'coreference':
        test(text_model, mention_model, image_model, coref_model, test_loader, args)
      elif task == 'retrieval':
        test_retrieve(text_model, image_model, coref_model, test_loader, args)
      elif task == 'both':
        test_retrieve(text_model, image_model, coref_model, test_loader, args)
        test(text_model, mention_model, image_model, coref_model, test_loader, args)
  if args.evaluate_only:
    task = config.get('task', 'coreference')
    print(task)
    task = config.get('task', 'coreference')
    if task == 'coreference':
      test(text_model, mention_model, image_model, coref_model, test_loader, args)
    elif task == 'retrieval':
      test_retrieve(text_model, image_model, coref_model, test_loader, args)
    elif task == 'both':
      test_retrieve(text_model, image_model, coref_model, test_loader, args)
      test(text_model, mention_model, image_model, coref_model, test_loader, args)

def test(text_model, mention_model, image_model, coref_model, test_loader, args):
    config = pyhocon.ConfigFactory.parse_file(args.config)
    documents = test_loader.dataset.documents
    all_grounding_scores = []
    all_grounding_labels = []
    all_text_scores = []
    all_text_labels = []
    text_model.eval()
    image_model.eval()
    coref_model.eval()
    with torch.no_grad(): 
      pred_text_dicts = []
      pred_grounding_dicts = []
      for i, batch in enumerate(test_loader):
        doc_embeddings, start_mappings, end_mappings, continuous_mappings,\
        width, videos,\
        text_labels, img_labels,\
        text_mask, span_mask, video_mask = batch   

        token_num = text_mask.sum(-1).long()
        span_num = span_mask.sum(-1).long()
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

        start_embeddings = torch.matmul(start_mappings, text_output)
        end_embeddings = torch.matmul(end_mappings, text_output)
        start_end_embeddings = torch.cat([start_embeddings, end_embeddings], dim=-1)
        continuous_embeddings = torch.matmul(continuous_mappings, text_output.unsqueeze(1))
        mention_output = mention_model(start_end_embeddings, continuous_embeddings, width)

        # Compute score for each span pair
        B = doc_embeddings.size(0) 
        for idx in range(B):
          grounded_text_scores, text_scores, grounding_scores = coref_model.module.predict(text_output[idx, :token_num[idx]], 
                                                         mention_output[idx, :span_num[idx]],
                                                         video_output[idx, :region_num[idx]],
                                                         continuous_mappings)
          first_grounding_idx, second_grounding_idx, pairwise_grounding_labels = get_pairwise_labels(text_labels[idx, :span_num[idx]].unsqueeze(0), 
                                                                                                     img_labels[idx, :region_num[idx]].unsqueeze(0), 
                                                                                                     is_training=False, device=device)
          first_text_idx, second_text_idx, pairwise_text_labels = get_pairwise_text_labels(text_labels[idx, :span_num[idx]].unsqueeze(0), 
                                                                                           is_training=False, device=device)
          # TODO clusters, scores = coref_model.module.predict_cluster(text_output[idx], video_output[idx],\
          #                                                      first_idx, second_idx)
          all_text_scores.append(text_scores)
          all_text_labels.append(pairwise_text_labels.to(torch.int))
          all_grounding_scores.append(grounding_scores.flatten())             
          all_grounding_labels.append(pairwise_grounding_labels.flatten().to(torch.int)) 
          
          global_idx = i * test_loader.batch_size + idx
          doc_id = test_loader.dataset.doc_ids[global_idx] 
          origin_tokens = [token[2] for token in test_loader.dataset.origin_tokens[global_idx]]
          candidate_start_ends = test_loader.dataset.origin_candidate_start_ends[global_idx]
          image_labels = test_loader.dataset.image_labels[global_idx]
          # print(doc_id, clusters.values(), candidate_start_ends.tolist())
          # TODO doc_name = doc_id
          document = {doc_id:test_loader.dataset.documents[doc_id]}
          # write_output_file(document, clusters, [doc_id]*candidate_start_ends.shape[0],
          #                  candidate_start_ends[:, 0].tolist(),
          #                  candidate_start_ends[:, 1].tolist(),
          #                  os.path.join(config['model_path'], 'pred_conll'),
          #                  doc_name, False, True)
          pred_text_dicts.append({'doc_id': doc_id,
                             'first_idx': first_text_idx.cpu().detach().numpy().tolist(),
                             'second_idx': second_text_idx.cpu().detach().numpy().tolist(),
                             'tokens': origin_tokens,
                             'mention_spans': candidate_start_ends.tolist(),
                             'score': text_scores.flatten().cpu().detach().numpy().tolist(),
                             'pairwise_label': pairwise_text_labels.cpu().detach().numpy().tolist()})

          pred_grounding_dicts.append({'doc_id': doc_id,
                             'first_idx': first_grounding_idx.cpu().detach().numpy().tolist(),
                             'second_idx': second_grounding_idx.cpu().detach().numpy().tolist(),
                             'tokens': origin_tokens,
                             'mention_spans': candidate_start_ends.tolist(),
                             'image_labels': image_labels,
                             'score': grounding_scores.flatten().cpu().detach().numpy().tolist(),
                             'pairwise_label': pairwise_grounding_labels.cpu().detach().numpy().tolist()})
      all_grounding_scores = torch.cat(all_grounding_scores)
      all_grounding_labels = torch.cat(all_grounding_labels)
      all_text_scores = torch.cat(all_text_scores)
      all_text_labels = torch.cat(all_text_labels)
      
      strict_grounding_preds = (all_grounding_scores > 0).to(torch.int)
      eval = Evaluation(strict_grounding_preds, all_grounding_labels.to(device))
      print('Number of crossmedia predictions: {}/{}'.format(strict_grounding_preds.sum(), len(strict_grounding_preds)))
      print('Number of crossmedia positive pairs: {}/{}'.format(len((all_grounding_labels == 1).nonzero()),
                                                     len(all_grounding_labels)))
      print('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                eval.get_precision(), eval.get_f1()))
      logger.info('Number of crossmedia predictions: {}/{}'.format(strict_grounding_preds.sum(), len(strict_grounding_preds)))
      logger.info('Number of crossmedia positive pairs: {}/{}'.format(len((all_grounding_labels == 1).nonzero()),
                                                           len(all_grounding_labels)))
      logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                      eval.get_precision(), eval.get_f1()))

      strict_text_preds = (all_text_scores > 0).to(torch.int)
      eval = Evaluation(strict_text_preds, all_text_labels.to(device))
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
      json.dump(pred_text_dicts, open(os.path.join(args.exp_dir, '{}_prediction_text_coref.json'.format(args.config.split('.')[0].split('/')[-1])), 'w'), indent=4)
      json.dump(pred_grounding_dicts, open(os.path.join(args.exp_dir, '{}_prediction_crossmedia_coref.json'.format(args.config.split('.')[0].split('/')[-1])), 'w'), indent=4)

def test_retrieve(text_model, image_model, coref_model, test_loader, args):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  text_embeddings = []
  video_embeddings = []
  text_masks = []
  video_masks = []
  text_model.eval()
  image_model.eval()
  coref_model.eval()
  with torch.no_grad():
    pred_dicts = []
    for i, batch in enumerate(test_loader):
      doc_embeddings, start_mappings, end_mappings, continuous_mappings,\
      width, videos,\
      text_labels, img_labels,\
      text_mask, span_mask, video_mask = batch   

      token_num = text_mask.sum(-1).long()
      span_num = span_mask.sum(-1).long()
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

    I2S_idxs, S2I_idxs = coref_model.module.retrieve(text_embeddings, video_embeddings, text_masks, video_masks)
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
    logger.info('Number of article-video pairs: {}'.format(I2S_idxs.size(0)))
    logger.info('I2S recall@1={}\tI2S recall@5={}\tI2S recall@10={}'.format(I2S_r1, I2S_r5, I2S_r10)) 
    logger.info('S2I recall@1={}\tS2I recall@5={}\tS2I recall@10={}'.format(S2I_r1, S2I_r5, S2I_r10)) 



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
  
  pred_out_dir = os.path.join(config['model_path'], 'pred_conll')
  if not os.path.isdir(pred_out_dir):
    os.mkdir(pred_out_dir)

  logging.basicConfig(filename=os.path.join(config['log_path'],'{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                      format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) 

  # Initialize dataloaders
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
                                                    config, split='train')
      test_set = SupervisedGroundingFeatureDataset(os.path.join(config['data_folder'], 'test.json'), 
                                                   os.path.join(config['data_folder'], 'test_mixed.json'), 
                                                   os.path.join(config['data_folder'], 'test_bboxes.json'), 
                                                   config, split='test')
 
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

  # Initialize models
  embedding_dim = config.bert_hidden_size # if config.with_head_attention else config.bert_hidden_size * 2
  assert embedding_dim == 2 * config.hidden_layer

  if config.get('glove_dimension', None):
    text_model = BiLSTM(config.glove_dimension, embedding_dim // 2) 
  else:
    text_model = BiLSTM(1024, embedding_dim // 2)


  if config['img_feat_type'] == 'resnet34':
    image_model = nn.Linear(512, embedding_dim)
  else: 
    image_model = nn.Linear(2048, embedding_dim)
  mention_model = SpanEmbedder(config)
  coref_model = JointSupervisedGroundedCoreferencer(config).to(device)

  if config['training_method'] in ('pipeline', 'continue'):
      text_model.load_state_dict(torch.load(config['span_repr_path'], map_location=device))
      image_model.load_state_dict(torch.load(config['image_repr_path'], map_location=device))
      coref_model.text_scorer.load_state_dict(torch.load(config['pairwise_scorer_path'], map_location=device))
  
  # Training
  train(text_model, mention_model, image_model, coref_model, train_loader, test_loader, args)
