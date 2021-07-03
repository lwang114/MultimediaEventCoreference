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
from text_models import BERTSpanEmbedder, SpanEmbedder, BiLSTM, SimplePairWiseClassifier
from visual_models import BiLSTMVideoEncoder, CrossmediaPairWiseClassifier
from corpus import TextVideoEventDataset
from evaluator import Evaluation, CoNLLEvaluation
from utils import make_prediction_readable, create_type_stoi, create_feature_stoi


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
          text_coref_model, 
          visual_coref_model,
          train_loader, 
          test_loader, 
          args, random_seed=None):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  if random_seed:
      config.random_seed = random_seed
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(True)
  
  if not isinstance(text_model, torch.nn.DataParallel):
    text_model = nn.DataParallel(text_model)

  if not isinstance(mention_model, torch.nn.DataParallel):
    mention_model = nn.DataParallel(mention_model)

  if not isinstance(visual_model, torch.nn.DataParallel):
    visual_model = nn.DataParallel(visual_model)
  
  if not isinstance(text_coref_model, torch.nn.DataParallel):
    text_coref_model = nn.DataParallel(text_coref_model)

  if not isinstance(visual_coref_model, torch.nn.DataParallel):
    visual_coref_model = nn.DataParallel(visual_coref_model)
 
  text_model.to(device)
  mention_model.to(device)
  visual_model.to(device)
  text_coref_model.to(device)
  visual_coref_model.to(device)

  # Create/load exp
  if args.start_epoch != 0:
    text_model.load_state_dict(torch.load('{}/text_model.pth'.format(config['model_path'], args.start_epoch)))
    visual_model.load_state_dict(torch.load('{}/visual_model.pth'.format(config['model_path'], args.start_epoch)))
    text_coref_model.load_state_dict(torch.load('{}/text_coref_model.{}.pth'.format(config['model_path'], args.start_epoch)))
    visual_coref_model.load_state_dict(torch.load('{}/visual_coref_scorer.{}.pth'.format(config['model_path'], args.start_epoch)))

  # Define the training criterion
  criterion = nn.BCEWithLogitsLoss()

  # Set up the optimizer  
  optimizer = get_optimizer(config, [text_model, 
                                     mention_model, 
                                     visual_model, 
                                     text_coref_model,
                                     visual_coref_model])
   
  # Start training
  total_loss = 0.
  total = 0.
  best_text_f1 = 0.
  best_grounding_f1 = 0.
  results = {}
  begin_time = time.time()
  if args.evaluate_only:
    config.epochs = 0

  for epoch in range(args.start_epoch, config.epochs):
    text_model.train()
    mention_model.train()
    visual_model.train()
    text_coref_model.train()
    visual_coref_model.train()
    for i, batch in enumerate(train_loader):
      doc_embeddings = batch['doc_embeddings'].to(device)
      start_mappings = batch['start_mappings'].to(device)
      end_mappings = batch['end_mappings'].to(device)
      continuous_mappings = batch['continuous_mappings'].to(device)
      width = batch['width'].to(device)
      start_arg_mappings = batch['start_arg_mappings'].to(device)
      end_arg_mappings = batch['end_arg_mappings'].to(device)
      continuous_arg_mappings = batch['continuous_arg_mappings'].to(device)
      arg_width = batch['arg_width'].to(device)
      n_args = batch['n_args'].to(device)

      videos = batch['action_embeddings'].to(device)
      text_labels = batch['cluster_labels'].to(device)
      event_labels = batch['event_labels'].to(device)
      event_linguistic_labels = torch.stack(
                            [batch['linguistic_labels'][feat_type].to(device)\
                             for feat_type in config.linguistic_feature_types],
                            dim=2)
      arg_linguistic_labels = torch.stack(
                          [batch['arg_linguistic_labels'][feat_type].to(device)\
                           for feat_type in config.linguistic_feature_types],
                          dim=2) 
      action_labels = batch['action_labels'].to(device) 
      text_mask = batch['text_mask'].to(device)
      span_mask = batch['span_mask'].to(device)
      action_mask = batch['action_mask'].to(device)
      span_num = torch.where(span_mask.sum(-1) > 0, 
                             torch.tensor(1, dtype=torch.int,
                                              device=doc_embeddings.device), 
                             torch.tensor(0, dtype=torch.int, 
                                              device=doc_embeddings.device)).sum(-1)
      action_num = torch.where(action_mask.sum(-1) > 0,
                             torch.tensor(1, dtype=torch.int,
                                              device=doc_embeddings.device), 
                             torch.tensor(0, dtype=torch.int, 
                                              device=doc_embeddings.device)).sum(-1)

      video_output = visual_model(videos, action_mask)
      mention_output = mention_model(doc_embeddings,
                                     start_mappings,
                                     end_mappings,
                                     continuous_mappings, 
                                     width,
                                     event_linguistic_labels,
                                     attention_mask=text_mask)
      argument_output = mention_model(doc_embeddings,
                                      start_arg_mappings,
                                      end_arg_mappings,
                                      continuous_arg_mappings,
                                      arg_width,
                                      arg_linguistic_labels,
                                      attention_mask=text_mask)
      crossmedia_mention_output = text_model(mention_output)
      text_scores = []
      video_scores = []
      pairwise_grounding_labels = []
      pairwise_text_labels = [] 
      B = doc_embeddings.size(0)
      for idx in range(B):
        first_text_idx,\
        second_text_idx,\
        pairwise_text_label = get_pairwise_text_labels(
                                 text_labels[idx, :span_num[idx]].unsqueeze(0),
                                 is_training=False,
                                 device=device)
        first_grounding_idx,\
        second_grounding_idx,\
        pairwise_grounding_label = get_pairwise_labels(
                                      event_labels[idx, :span_num[idx]],
                                      action_labels[idx, :action_num[idx]],
                                      is_training=False,
                                      device=device)
        if first_grounding_idx is None or first_text_idx is None:
          continue

        first_text_idx = first_text_idx.squeeze(0)
        second_text_idx = second_text_idx.squeeze(0)
        pairwise_text_label = pairwise_text_label.squeeze(0)
        n_pairs = first_text_idx.shape[0]

        video_score = visual_coref_model(crossmedia_mention_output[idx, first_grounding_idx],
                                         video_output[idx, second_grounding_idx]) 
        crossmedia_score = visual_coref_model.module.crossmedia_score(
                                                       first_text_idx,
                                                       second_text_idx,
                                                       video_score)
        text_score = text_coref_model(mention_output[idx, first_text_idx],
                                      mention_output[idx, second_text_idx])

        argument_output_3d = argument_output[idx, :span_num[idx]*n_args[idx]]\
                             .view(span_num[idx], n_args[idx], -1)
        
        argument_score = text_coref_model(argument_output_3d[first_text_idx]\
                                          .view(n_pairs*n_args[idx], -1),
                                          argument_output_3d[second_text_idx]\
                                          .view(n_pairs*n_args[idx], -1))
        argument_score = argument_score.view(n_pairs, n_args[idx]).mean(-1, keepdim=True)
        text_score = (text_score + crossmedia_score + argument_score) / 3.

        text_scores.append(text_score) 
        video_scores.append(video_score) 
        pairwise_text_labels.append(pairwise_text_label)
        pairwise_grounding_labels.append(pairwise_grounding_label)

      if not len(text_scores):
        continue

      text_scores = torch.cat(text_scores).squeeze(1)
      video_scores = torch.cat(video_scores).squeeze(1)
      pairwise_text_labels = torch.cat(pairwise_text_labels).to(torch.float)
      pairwise_grounding_labels = torch.cat(pairwise_grounding_labels).to(torch.float)

      text_loss = criterion(text_scores, pairwise_text_labels)
      video_loss = criterion(video_scores, pairwise_grounding_labels)
      loss = text_loss + video_loss

      optimizer.zero_grad()
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
    torch.save(visual_model.module.state_dict(), '{}/visual_model-{}.pth'.format(config['model_path'], config['random_seed']))
    torch.save(mention_model.module.state_dict(), '{}/mention_model-{}.pth'.format(config['model_path'], config['random_seed']))
    torch.save(text_coref_model.module.state_dict(), '{}/text_coref_model-{}.pth'.format(config['model_path'], config['random_seed']))
    torch.save(visual_coref_model.module.state_dict(), '{}/visual_coref_model-{}.pth'.format(config['model_path'], config['random_seed']))
 
    if epoch % 1 == 0:
      res = test(text_model, 
                 mention_model, 
                 visual_model, 
                 text_coref_model, 
                 visual_coref_model,
                 test_loader, args)
      if res['pairwise'][-1] >= best_text_f1:
        best_text_f1 = res['pairwise'][-1]
        results['pairwise'] = res['pairwise']
        results['muc'] = res['muc']
        results['ceafe'] = res['ceafe']
        results['bcubed'] = res['bcubed']
        results['avg'] = res['avg']
        torch.save(text_model.module.state_dict(), '{}/best_text_model-{}.pth'.format(config['model_path'], config['random_seed']))
        torch.save(visual_model.module.state_dict(), '{}/best_visual_model-{}.pth'.format(config['model_path'], config['random_seed']))
        torch.save(mention_model.module.state_dict(), '{}/best_mention_model-{}.pth'.format(config['model_path'], config['random_seed']))
        torch.save(text_coref_model.module.state_dict(), '{}/best_text_coref_model-{}.pth'.format(config['model_path'], config['random_seed']))
        torch.save(visual_coref_model.module.state_dict(), '{}/best_visual_coref_model-{}.pth'.format(config['model_path'], config['random_seed']))
        print('Best text coreference F1={}'.format(best_text_f1))

  if args.evaluate_only:
    results = test(text_model, mention_model, visual_model, text_coref_model, visual_coref_model, test_loader, args)
  return results
      
def test(text_model,
         mention_model, 
         visual_model, 
         text_coref_model,
         visual_coref_model,
         test_loader, 
         args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = pyhocon.ConfigFactory.parse_file(args.config)
    documents = test_loader.dataset.documents
    all_scores = []
    all_labels = []
    all_grounding_scores = []
    all_grounding_labels = []

    text_model.eval()
    visual_model.eval()
    text_coref_model.eval()
    visual_coref_model.eval()

    conll_eval = CoNLLEvaluation()
    f_out = open(os.path.join(config['model_path'], 'prediction.readable'), 'w')
    best_f1 = 0.
    results = {} 
    with torch.no_grad():     
        for i, batch in enumerate(test_loader):
          doc_embeddings = batch['doc_embeddings'].to(device)
          start_mappings = batch['start_mappings'].to(device)
          end_mappings = batch['end_mappings'].to(device)
          continuous_mappings = batch['continuous_mappings'].to(device)
          width = batch['width'].to(device)
          start_arg_mappings = batch['start_arg_mappings'].to(device)
          end_arg_mappings = batch['end_arg_mappings'].to(device)
          continuous_arg_mappings = batch['continuous_arg_mappings'].to(device)
          arg_width = batch['arg_width'].to(device)
          n_args = batch['n_args'].to(device)

          videos = batch['action_embeddings'].to(device)
          text_labels = batch['cluster_labels'].to(device) 
          event_labels = batch['event_labels'].to(device)
          event_linguistic_labels = torch.stack(
                                [batch['linguistic_labels'][feat_type].to(device)\
                                 for feat_type in config.linguistic_feature_types],
                                dim=2)
          arg_linguistic_labels = torch.stack(
                                [batch['arg_linguistic_labels'][feat_type].to(device)\
                                 for feat_type in config.linguistic_feature_types],
                                dim=2) # TODO Check dim
          action_labels = batch['action_labels'].to(device)
          text_mask = batch['text_mask'].to(device)
          span_mask = batch['span_mask'].to(device)
          action_mask = batch['action_mask'].to(device)
          
          token_num = text_mask.sum(-1).long()
          span_num = torch.where(span_mask.sum(-1) > 0, 
                                 torch.tensor(1, dtype=torch.int,
                                              device=doc_embeddings.device), 
                                 torch.tensor(0, dtype=torch.int, 
                                              device=doc_embeddings.device)).sum(-1)
          action_num = torch.where(action_mask.sum(-1) > 0,
                                 torch.tensor(1, dtype=torch.int,
                                              device=doc_embeddings.device), 
                                 torch.tensor(0, dtype=torch.int, 
                                              device=doc_embeddings.device)).sum(-1)

          
          # Extract span and video embeddings
          video_output = visual_model(videos, action_mask)
          mention_output = mention_model(doc_embeddings,
                                         start_mappings, 
                                         end_mappings,
                                         continuous_mappings, 
                                         width,
                                         event_linguistic_labels,
                                         attention_mask=text_mask)
          argument_output = mention_model(doc_embeddings,
                                          start_arg_mappings,
                                          end_arg_mappings,
                                          continuous_arg_mappings,
                                          arg_width,
                                          arg_linguistic_labels,
                                          attention_mask=text_mask)
          crossmedia_mention_output = text_model(mention_output)

          B = doc_embeddings.size(0) 
          for idx in range(B):
            global_idx = i * test_loader.batch_size + idx

            first_text_idx,\
            second_text_idx,\
            pairwise_text_labels = get_pairwise_text_labels(
                                       text_labels[idx, :span_num[idx]].unsqueeze(0), 
                                       is_training=False,
                                       device=device)
            first_grounding_idx,\
            second_grounding_idx,\
            pairwise_grounding_labels = get_pairwise_labels(
                                           event_labels[idx, :span_num[idx]],
                                           action_labels[idx, :action_num[idx]],
                                           is_training=False,
                                           device=device)

            if first_text_idx is None or first_grounding_idx is None:
                continue
        
            first_text_idx = first_text_idx.squeeze(0)
            second_text_idx = second_text_idx.squeeze(0)
            pairwise_text_labels = pairwise_text_labels.squeeze(0)
            n_pairs = first_text_idx.shape[0]
            text_score = text_coref_model(mention_output[idx, first_text_idx],
                                          mention_output[idx, second_text_idx])

            visual_scores = visual_coref_model(crossmedia_mention_output[idx, first_grounding_idx],
                                               video_output[idx, second_grounding_idx])
            crossmedia_scores = visual_coref_model.module.crossmedia_score(
                                                            first_text_idx,
                                                            second_text_idx,
                                                            visual_scores)
            text_scores = text_coref_model(mention_output[idx, first_text_idx],
                                           mention_output[idx, second_text_idx])
            argument_output_3d = argument_output[idx, :span_num[idx]*n_args[idx]]\
                                 .view(span_num[idx], n_args[idx], -1)
            argument_scores = text_coref_model(argument_output_3d[first_text_idx]\
                                                   .view(n_pairs*n_args[idx], -1),
                                               argument_output_3d[second_text_idx]\
                                                   .view(n_pairs*n_args[idx], -1))
            argument_scores = argument_scores.view(n_pairs, n_args[idx]).mean(-1, keepdim=True)
            text_scores = (text_scores + crossmedia_scores + argument_scores) / 3.
            predicted_antecedents = text_coref_model.module.predict_cluster(
                                               text_scores, 
                                               first_text_idx,
                                               second_text_idx) 
            origin_candidate_start_ends = test_loader.dataset.origin_candidate_start_ends[global_idx]
            predicted_antecedents = torch.LongTensor(predicted_antecedents)
            origin_candidate_start_ends = torch.LongTensor(origin_candidate_start_ends)
            
            pred_clusters, gold_clusters = conll_eval(origin_candidate_start_ends,
                                                      predicted_antecedents,
                                                      origin_candidate_start_ends,
                                                      text_labels[idx, :span_num[idx]])
            doc_id = test_loader.dataset.doc_ids[global_idx]
            tokens = [token[2] for token in test_loader.dataset.documents[doc_id]]
            event_label_dict = test_loader.dataset.event_label_dict[doc_id]
            arg_spans = test_loader.dataset.origin_argument_spans[global_idx]
            arguments = {span: arg_span\
                         for span, arg_span in zip(sorted(event_label_dict), arg_spans)}
            pred_clusters_str,\
            gold_clusters_str = conll_eval.make_output_readable(
                                  pred_clusters, 
                                  gold_clusters,
                                  tokens, arguments=arguments
                                )
            token_str = ' '.join(tokens).replace('\n', '')
            f_out.write(f"{doc_id}: {token_str}\n")
            f_out.write(f'Pred: {pred_clusters_str}\n')
            f_out.write(f'Gold: {gold_clusters_str}\n\n')

            all_scores.append(text_scores.squeeze(1))
            all_labels.append(pairwise_text_labels.to(torch.int))
            all_grounding_scores.append(visual_scores.squeeze(1))
            all_grounding_labels.append(pairwise_grounding_labels.to(torch.int)) 
        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        all_grounding_scores = torch.cat(all_grounding_scores)
        all_grounding_labels = torch.cat(all_grounding_labels)

        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))
        print('[Text Coreference Result]')
        print('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        print('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                       len(all_labels)))
        print('Pairwise - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                    eval.get_precision(), 
                                                                    eval.get_f1()))
        strict_preds = (all_grounding_scores > 0).to(torch.int)
        grounding_eval = Evaluation(strict_preds, all_grounding_labels.to(device))
        print('[Crossmedia Coreference Result]')
        print('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        print('Number of positive pairs: {}/{}'.format(len((all_grounding_labels == 1).nonzero()),
                                                       len(all_grounding_labels)))
        print('Pairwise - Recall: {}, Precision: {}, F1: {}'.format(grounding_eval.get_recall(),
                                                                    grounding_eval.get_precision(), 
                                                                    grounding_eval.get_f1()))

        muc, b_cubed, ceafe, avg = conll_eval.get_metrics()
        results['pairwise'] = (eval.get_precision().item(), eval.get_recall().item(), eval.get_f1().item())
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
  # Set up argument parser
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='configs/config_coref_simple_video_m2e2.json')
  parser.add_argument('--start_epoch', type=int, default=0)
  parser.add_argument('--evaluate_only', action='store_true')
  parser.add_argument('--compute_confidence_bound', action='store_true')
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
  splits = [os.path.join(config['data_folder'], 'train_events.json'),\
            os.path.join(config['data_folder'], 'test_events.json'),\
            os.path.join(config['data_folder'], 'train_entities.json'),\
            os.path.join(config['data_folder'], 'test_entities.json')]
  
  event_stoi = create_type_stoi(splits) 
  feature_stoi = create_feature_stoi(splits, feature_types=config['linguistic_feature_types'])
 
  train_set = TextVideoEventDataset(config, 
                                    event_stoi, 
                                    feature_stoi,
                                    split='train')
  test_set = TextVideoEventDataset(config, 
                                   event_stoi, 
                                   feature_stoi,
                                   split='test')

  pairwises  = []
  mucs = []
  bcubeds = []
  ceafes = []
  avgs = []

  if args.compute_confidence_bound:
      seeds = [1111, 2222, 3333, 4444]
  else:
      seeds = [1111]
      
  for seed in seeds:
      config.random_seed = seed
      config['random_seed'] = seed
      fix_seed(config)
  
      # Initialize dataloaders 
      train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
      test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

      # Initialize models
      if config.get('finetune_bert', False):
        mention_model = BERTSpanEmbedder(config, device)
      else:
        mention_model = SpanEmbedder(config, device)
      text_coref_model = SimplePairWiseClassifier(config).to(device)
      text_model = BiLSTM(int(text_coref_model.input_layer // 3),
                          int(config.hidden_layer // 2))
      visual_model = BiLSTMVideoEncoder(400, int(config.hidden_layer // 2))
      visual_coref_model = CrossmediaPairWiseClassifier(config).to(device)
      
      if config['training_method'] in ('pipeline', 'continue') or args.evaluate_only:
          text_model.load_state_dict(torch.load(config['text_model_path'], map_location=device))
          for p in text_model.parameters():
              p.requires_grad = False
          
          mention_model.load_state_dict(torch.load(config['mention_model_path']))
          for p in mention_model.parameters():
              p.requires_grad = False
          
          text_coref_model.load_state_dict(torch.load(config['text_coref_model_path'], map_location=device))
          for p in text_coref_model.parameters():
              p.requires_grad = False
          
          visual_coref_model.load_state_dict(torch.load(config['visual_coref_model_path'], map_location=device))
          for p in visual_coref_model.parameters():
              p.requires_grad = True

      # Training
      n_params = 0
      for p in text_model.parameters():
          n_params += p.numel()

      for p in mention_model.parameters():
          n_params += p.numel()

      for p in text_coref_model.parameters():
          n_params += p.numel()

      for p in visual_coref_model.parameters():
          n_params += p.numel()

      print('Number of parameters in coref classifier: {}'.format(n_params))
      results = train(text_model, 
                      mention_model, 
                      visual_model, 
                      text_coref_model, 
                      visual_coref_model,
                      train_loader, 
                      test_loader, 
                      args, random_seed=seed)
      pairwises.append(results['pairwise'])
      mucs.append(results['muc'])
      bcubeds.append(results['bcubed'])
      ceafes.append(results['ceafe'])
      avgs.append(results['avg'])

  mean_pairwise, std_pairwise = np.mean(np.asarray(pairwises), axis=0), np.std(np.asarray(pairwises), axis=0)
  mean_muc, std_muc = np.mean(np.asarray(mucs), axis=0), np.std(np.asarray(mucs), axis=0)
  mean_bcubed, std_bcubed = np.mean(np.asarray(bcubeds), axis=0), np.std(np.asarray(bcubeds), axis=0)
  mean_ceafe, std_ceafe = np.mean(np.asarray(ceafes), axis=0), np.std(np.asarray(ceafes), axis=0)
  mean_avg, std_avg = np.mean(np.asarray(avgs), axis=0), np.std(np.asarray(avgs), axis=0)
  print(f'Pairwise: precision {mean_pairwise[0]} +/- {std_pairwise[0]}, '
        f'recall {mean_pairwise[1]} +/- {std_pairwise[1]}, '
        f'f1 {mean_pairwise[2]} +/- {std_pairwise[2]}')
  print(f'MUC: precision {mean_muc[0]} +/- {std_muc[0]}, '
        f'recall {mean_muc[1]} +/- {std_muc[1]}, '
        f'f1 {mean_muc[2]} +/- {std_muc[2]}')
  print(f'Bcubed: precision {mean_bcubed[0]} +/- {std_bcubed[0]}, '
        f'recall {mean_bcubed[1]} +/- {std_bcubed[1]}, '
        f'f1 {mean_bcubed[2]} +/- {std_bcubed[2]}')
  print(f'CEAFe: precision {mean_ceafe[0]} +/- {std_ceafe[0]}, '
        f'recall {mean_ceafe[1]} +/- {std_ceafe[1]}, '
        f'f1 {mean_ceafe[2]} +/- {std_ceafe[2]}')
  print(f'CoNLL: precision {mean_avg[0]} +/- {std_avg[0]}, '
        f'recall {mean_avg[1]} +/- {std_avg[1]}, '
        f'f1 {mean_avg[2]} +/- {std_avg[2]}')
