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
from text_models import SpanEmbedder, BiLSTM, SimplePairWiseClassifier
from corpus import SupervisedGroundingFeatureDataset
from evaluator import Evaluation, CoNLLEvaluation
from sklearn.metrics import average_precision_score

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

def train(text_model, visual_model, crossmedia_model, coref_model, train_loader, test_loader, args, random_seed=None):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  if random_seed:
      config.random_seed = random_seed
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(True)
  
  if not isinstance(text_model, torch.nn.DataParallel):
    text_model = nn.DataParallel(text_model)

  if not isinstance(visual_model, torch.nn.DataParallel):
    visual_model = nn.DataParallel(visual_model)
  
  if not isinstance(crossmedia_model, torch.nn.DataParallel):
    crossmedia_model = nn.DataParallel(crossmedia_model)

  if not isinstance(coref_model, torch.nn.DataParallel):
    coref_model = nn.DataParallel(coref_model)

  text_model.to(device)
  mention_model.to(device)
  visual_model.to(device)
  crossmedia_model.to(device)
  coref_model.to(device)

  # Define the training criterion
  criterion = nn.BCEWithLogitsLoss()

  # Set up the optimizer  
  optimizer = get_optimizer(config, [text_model, coref_model])
   
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
  for epoch in range(config.epochs):
    text_model.train()
    visual_model.train()
    crossmedia_model.train()
    coref_model.train()
    for i, batch in enumerate(train_loader):
      doc_embeddings, start_mappings, end_mappings, continuous_mappings,\
      width, action_embeddings,\
      text_labels, img_labels,\
      text_mask, span_mask, action_mask = batch   

      B = doc_embeddings.size(0)     
      doc_embeddings = doc_embeddings.to(device)
      start_mappings = start_mappings.to(device)
      end_mappings = end_mappings.to(device)

      action_embeddings = action_embeddings.to(device)
      text_labels = text_labels.to(device)
      text_mask = text_mask.to(device)
      span_mask = span_mask.to(device)
      action_mask = action_mask.to(device)

      first_text_idx, second_text_idx, pairwise_text_labels = get_pairwise_text_labels(text_labels, is_training=False, device=device)      
      pairwise_text_labels = pairwise_text_labels.to(torch.float).flatten()
      optimizer.zero_grad()

      action_output = visual_model(action_embeddings).sum(1) / action_mask.sum(-1).unsqueeze(-1)
      text_output = text_model(doc_embeddings)
      mention_start_output = torch.matmul(start_mappings, text_output)
      mention_end_output = torch.matmul(end_mappings, text_output)
      mention_output = (mention_start_output + mention_end_output) / 2.  
      crossmedia_output = crossmedia_model(mention_output)
      align_output = align(crossmedia_output, action_output)
      mention_output = torch.cat([mention_output, align_output], dim=-1)

      scores = []
      for idx in range(B):
          scores.append(coref_model(mention_output[idx, first_text_idx[idx]],
                                    mention_output[idx, second_text_idx[idx]]))
      scores = torch.cat(scores).squeeze(1)
      loss = criterion(scores, pairwise_text_labels)
      
      loss.backward()
      optimizer.step()

      total_loss += loss.item() * B
      total += B
      if i % 200 == 0:
        info = 'Iter {} {:.4f}'.format(i, total_loss / total)
        print(info)
        logging.info(info) 
    
    info = 'Epoch: [{}][{}/{}]\tTime {:.3f}\tLoss total {:.4f} ({:.4f})'.format(epoch, i, len(train_loader), time.time()-begin_time, total_loss, total_loss / total)
    print(info)
    logging.info(info)

    torch.save(text_model.module.state_dict(), '{}/text_model-{}.pth'.format(config['model_path'], config['random_seed']))
    torch.save(visual_model.module.state_dict(), '{}/visual_model-{}.pth'.format(config['model_path'], config['random_seed']))
    torch.save(coref_model.module.state_dict(), '{}/coref_model-{}.pth'.format(config['model_path'], config['random_seed']))
 
    if epoch % 1 == 0:
      res = test(text_model, visual_model, coref_model, test_loader, args)
      if res['pairwise'][-1] >= best_text_f1:
        best_text_f1 = res['pairwise'][-1]
        results['pairwise'] = res['pairwise']
        results['muc'] = res['muc']
        results['ceafe'] = res['ceafe']
        results['bcubed'] = res['bcubed']
        results['avg'] = res['avg']
        torch.save(text_model.module.state_dict(), '{}/best_text_model-{}.pth'.format(config['model_path'], config['random_seed']))
        torch.save(visual_model.module.state_dict(), '{}/best_visual_model-{}.pth'.format(config['model_path'], config['random_seed']))
        torch.save(coref_model.module.state_dict(), '{}/best_coref_model-{}.pth'.format(config['model_path'], config['random_seed']))
        print('Best text coreference F1={}'.format(best_text_f1))

  if args.evaluate_only:
    results = test(text_model, mention_model, visual_model, coref_model, test_loader, args)
  return results
      
def test(text_model, visual_model, coref_model, test_loader, args): # TODO 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = pyhocon.ConfigFactory.parse_file(args.config)
    documents = test_loader.dataset.documents
    all_scores = []
    all_labels = []

    text_model.eval()
    visual_model.eval()
    coref_model.eval()

    conll_eval = CoNLLEvaluation()
    f_out = open(os.path.join(config['model_path'], 'prediction.readable'), 'w')
    best_f1 = 0.
    results = {} 
    with torch.no_grad():     
        for i, batch in enumerate(test_loader):
          doc_embeddings,\
          start_mappings, end_mappings,\
          continuous_mappings,\
          width,\
          action_embeddings,\
          text_labels, type_labels, img_labels,\
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
          action_embeddings = videos.to(device)
          text_labels = text_labels.to(device)
          type_labels = type_labels.to(device)
          img_labels = img_labels.to(device)
          text_mask = text_mask.to(device)
          span_mask = span_mask.to(device)
          video_mask = video_mask.to(device)

          # Extract span and video embeddings
          action_output = visual_model(action_embeddings)
          text_output = text_model(doc_embeddings)
          mention_start_output = torch.matmul(start_mappings, text_output)
          mention_end_output = torch.matmul(end_mappings, text_output)
          mention_output = (mention_start_output + mention_end_output) / 2.
          crossmedia_output = crossmedia_model(mention_output)
          align_output = align(crossmedia_output, action_output)
          mention_output = torch.cat([mention_output, align_output], dim=-1)
         
          B = doc_embeddings.size(0) 
          for idx in range(B):
            global_idx = i * test_loader.batch_size + idx

            first_text_idx, second_text_idx, pairwise_text_labels = get_pairwise_text_labels(text_labels[idx, :span_num[idx]].unsqueeze(0), 
                                                                                             is_training=False, device=device)
            if first_text_idx is None:
                continue
            first_text_idx = first_text_idx.squeeze(0)
            second_text_idx = second_text_idx.squeeze(0)
            pairwise_text_labels = pairwise_text_labels.squeeze(0)
            predicted_antecedents, text_scores = coref_model.module.predict_cluster(mention_output[idx, :span_num[idx]], first_text_idx,
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
            pred_clusters_str, gold_clusters_str = conll_eval.make_output_readable(pred_clusters, gold_clusters, tokens)
            token_str = ' '.join(tokens).replace('\n', '')
            f_out.write(f"{doc_id}: {token_str}\n")
            f_out.write(f'Pred: {pred_clusters_str}\n')
            f_out.write(f'Gold: {gold_clusters_str}\n\n')

            all_scores.append(text_scores.squeeze(1))
            all_labels.append(pairwise_text_labels.to(torch.int).cpu())            
        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        # Compute mAP
        average_precision = average_precision_score(all_labels.cpu().detach().numpy(),
                                                    torch.sigmoid(all_scores).cpu().detach().numpy())

        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))

        print('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        print('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                     len(all_labels)))
        print('Pairwise - Recall: {}, Precision: {}, F1: {}, mAP: {}'.format(eval.get_recall(),
                                                                eval.get_precision(), eval.get_f1(), average_precision))
        
        muc, b_cubed, ceafe, avg = conll_eval.get_metrics()
        results['pairwise'] = (eval.get_precision().item(), eval.get_recall().item(), eval.get_f1().item(), average_precision)
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

def align(crossmedia_output, action_output):
  """
  :param crossmedia_output: (batch size, num. of spans, embedding dim), mention embedding mapped to the multimedia space
  :param action_output: (batch size, num. of actions, embedding dim), action embedding 
  :returns aligned output: (batch size, num. of spans, embedding dim)
  """
  batch_size = crossmedia_output.size(0)
  num_spans = crossmedia_output.size(1)
  num_actions = action_output.size(1)
  scores = torch.bmm(crossmedia_output, action_output.permute(0, 2, 1))
  alignment_mask = torch.zeros((batch_size, num_spans, num_actions))
  for i in range(num_spans):
    for j in range(num_actions):
      if scores[i, j] > 0:
        alignment_mask[i, j] = 1.
  alignment_mask /= torch.maximum(alignment_mask.sum(-1).unsqueeze(-1), 1.) 
  aligned_output = torch.bmm(alignment_mask, action_output)
  return aligned_output 


if __name__ == '__main__':
  # Set up argument parser
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='configs/config_coref_simple_video_m2e2.json')
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
  splits = [os.path.join(config['data_folder'], 'train_mixed.json'),\
            os.path.join(config['data_folder'], 'test_mixed.json')]
  type_to_idx = create_type_to_idx(splits) 
  role_to_idx = create_role_to_idx(splits)
 
  train_set = SupervisedGroundingFeatureDataset(os.path.join(config['data_folder'], 'train.json'), 
                                                os.path.join(config['data_folder'], f'train_{config.mention_type}.json'), 
                                                os.path.join(config['data_folder'], 'train_bboxes.json'),
                                                config, split='train', type_to_idx=type_to_idx, role_to_idx=role_to_idx)
  test_set = SupervisedGroundingFeatureDataset(os.path.join(config['data_folder'], 'test.json'),
                                               os.path.join(config['data_folder'], f'test_{config.mention_type}.json'), 
                                               os.path.join(config['data_folder'], 'test_bboxes.json'), 
                                               config, split='test', type_to_idx=type_to_idx, role_to_idx=role_to_idx)

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
      crossmedia_model = nn.Sequential(
                          nn.Linear(1024, 800),
                          nn.ReLU(),
                          nn.Linear(800, 800),
                          nn.ReLU(),
                          nn.Linear(800, 800),
                        )
      crossmedia_model.load_state_dict(torch.load(config['crossmedia_model_path'], map_location=device))
      visual_model = BiLSTM(400, 400, num_layers=3) # TODO Add config file
      visual_model.load_state_dict(torch.load(config['visual_model_path'], map_location=device))
      text_model = NoOp()
      coref_model = SimplePairWiseClassifier(config).to(device)
      
      if config['training_method'] in ('pipeline', 'continue'):
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
      results = train(text_model, mention_model, visual_model, coref_model, train_loader, test_loader, args, random_seed=seed)
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
        f'f1 {mean_pairwise[2]} +/- {std_pairwise[2]}, '
        f'mAP {mean_pairwise[3]} +/- {std_pairwise[3]}')
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
