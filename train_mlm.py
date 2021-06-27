import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
import pyhocon
import datetime
from itertools import combinations 
from corpus_masked_lm import TextEventMaskedLMDataset
from evaluator import Evaluation, CoNLLEvaluation
from transformers import (
  AutoModel,
  AdamW, 
  get_linear_schedule_with_warmup
)

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

def cluster_to_pairwise(labels):
  n = len(labels)
  if n <= 1:
    return None
  first, second = zip(*list(combinations(range(n), 2)))
  first = list(first)
  second = list(second)
  return (labels[first] != 0) & (labels[second] != 0) & (labels[first] == labels[second])

def predict_antecedents(scores):
  antecedents = -1 * torch.ones(span_num[idx], dtype=torch.long)
  cluster_labels = []
  num_spans = scores.size(0)
  num_clusters = 1
  for idx in range(num_spans):
    if idx == 0:
      clusters.append(num_clusters)
      continue 
    else:
      a_idx = scores[idx, :idx+1].max(1)[1] - 1
      if a_idx == -1:
        num_clusters += 1
        cluster_labels.append(num_clusters)
      else:
        antecedents[idx] = a_idx
        cluster_labels.append(cluster_labels[a_idx])
  cluster_labels = torch.LongTensor(cluster_labels)
  return antecedents, cluster_labels

def train(event_model,
          train_loader,
          test_loader,
          args, 
          random_seed=None):
  config = pyhocon.ConfigFactory.parse_file(args.config)
  if random_seed:
    config.random_seed = random_seed
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Data parallel
  if not isinstance(event_model, torch.nn.DataParallel):
    event_model = nn.DataParallel(event_model) 
  
  # Define training criterion
  criterion = nn.CrossEntropyLoss()

  # Set up optimizer
  optimizer = get_optimizer(config, [event_model])
  
  # Start training
  total_loss = 0.
  total = 0. 
  for epoch in range(config.epochs):
    event_model.train()
    for b_idx, batch in enumerate(train_loader):
      input_ids = batch['input_ids'].to(device)
      event_mask_input_ids = batch['event_mask_input_ids'].to(device)
      start_mappings = batch['start_mapping'].to(device) 
      end_mappings = batch['end_mapping'].to(device)
      continuous_mappings = batch['continuous_mappings'].to(device)
      width = batch['width'].to(device)

      # (batch size, max num. spans)
      event_ids = torch.matmul(input_ids, start_mappings.permute(0, 2, 1))[1:]
      # (batch size, max num. spans, max num. spans + 1)       
      masked_event_ids = torch.matmul(event_mask_input_ids,
                                      start_mappings.permute(0, 2, 1))
           
      # (batch size,)
      span_num = where(start_mappings.sum(-1) > 0, 
                       torch.tensor(1, dtype=torch.int),
                       torch.tensor(0, dtype=torch.int)).sum(1) 

      B = masked_event_ids.size(0)
      scores = []
      labels = []
      for idx in range(B):
        # (max num. spans + 1, vocab. size)
        logits = event_model(masked_event_ids[idx])[0]
        scores.extend([logits[span_idx, span_idx+1] for span_idx in range(span_num[idx])])
        labels.append(event_ids[:span_num[idx]])
      scores = torch.cat(logits)
      labels = torch.cat(labels)
      loss = criterion(scores, labels)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item() * B
      total += B
      if b_idx % 200 == 0:
        info = 'Iter {} {:.4f}'.format(b_idx, total_loss / total)
        print(info)
        logging.info(info)

    info = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tTime {:.3f}\tLoss total {total_loss:.4f} ({total_loss / total:.4f})'
    print(info)
    logging.info(info)
    
    # Save model
    torch.save(event_model.module.state_dict(), '{config["model_path"]}/text_model-{config["random_seed"]}.pth')
    if epoch % 1 == 0:
      res = test(event_model, test_loader, args)
  if args.evaluate_only:
    results = test(event_model)
  return results

def test(event_model, test_loader, args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  config = pyhocon.ConfigFactory.parse_file(args.config)
  documents = test_loader.dataset.documents
  
  pred_token_ids = [] # Predicted IDs for MLM
  gold_token_ids = [] # Gold IDs for MLM
  pred_pairwise_labels = [] # Predicted coreference labels
  gold_pairwise_labels = [] # Gold coreference labels
  
  conll_eval = CoNLLEvaluation()
  f_out = open(os.path.join(config['model_path'], 'prediction.readable'), 'w')
  best_coref_f1 = 0.
  best_masked_lm_acc = 0.
  results = {}
  with torch.no_grad():
    for b_idx, batch in enumerate(test_loader):
      input_ids = batch['input_ids'].to(device)
      event_mask_input_ids = batch['event_mask_input_ids'].to(device)
      start_mappings = batch['start_mapping'].to(device)
      end_mappings = batch['end_mapping'].to(device)
      width = batch['width'].to(device)
      cluster_labels = batch['cluster_labels'].to(device)

      # (batch size, max num. spans)
      event_ids = torch.matmul(input_ids, start_mappings.permute(0, 2, 1))[1:]
      # (batch size, max num. spans, max num. spans + 1)       
      masked_event_ids = torch.matmul(event_mask_input_ids,
                                      start_mappings.permute(0, 2, 1))
      
      # (batch size,)
      span_num = where(start_mappings.sum(-1) > 0,
                       torch.tensor(1, dtype=torch.int),
                       torch.tensor(0, dtype=torch.int)).sum(1)
      B = masked_event_ids.size(0)
      for idx in range(B):
        global_idx = b_idx * test_loader.batch_size + idx 
        # (max num. spans + 1, vocab. size)
        logits, attentions = event_model(masked_event_ids[idx], output_attentions=True)
        scores = torch.cat([logits[span_idx, span_idx+1] for span_idx in range(span_num[idx])])
        pred_token_ids.append(scores.max(-1)[1])
        gold_token_ids.append(input_ids[idx, :span_num[idx]])
      
        # Antecedent prediction
        pred_antecedents, pred_cluster_labels = predict_antecedents(attentions)        
         
        event_spans = test_loader.dataset.event_spans[global_idx]
        pred_clusters, gold_clusters = conll_eval(event_spans,
                                                  pred_antecedents,
                                                  event_spans,
                                                  cluster_labels[idx, :span_num[idx]])
        doc_id = test_loader.dataset.doc_ids[global_idx]
        tokens = [x[2] for x in test_loader.dataset.documents[doc_id]]
        pred_clusters_str,\
        gold_clusters_str = conll_eval.make_output_readable(
                              pred_clusters,
                              gold_clusters,
                              tokens)

        # Extract pairwise labels from antecedent
        pred_pairwise_labels.append(cluster_to_pairwise(pred_cluster_labels))
        gold_pairwise_labels.append(cluster_to_pairwise(cluster_labels[idx, :span_num[idx]]))

    pred_token_ids = torch.cat(pred_token_ids)
    gold_token_ids = torch.cat(gold_token_ids)
    n = pred_token_ids.size(0)
    correct = (gold_token_ids == pred_token_ids).float().sum()
    masked_lm_acc = correct / n

    pred_pairwise_labels = torch.cat(pred_pairwise_labels)
    gold_pairwise_labels = torch.cat(gold_pairwise_labels)
    coref_eval = Evaluation(pred_pairwise_labels, gold_pairwise_labels)

    print('[Masked Language Modeling Result]')
    print(f'Prediction accuracy = {masked_lm_acc:.2f}')
    print('[Text Coreference Result]')
    print('Number of predictions: {}/{}'.format(pred_pairwise_labels.sum(), len(pred_pairwise_labels)))
    print('Number of positive pairs: {}/{}'.format(len((gold_pairwise_labels == 1).nonzero()),
                                                   len(gold_pairwise_labels)))
    print('Pairwise - Recall: {}, Precision: {}, F1: {}'.format(coref_eval.get_recall(),
                                                                coref_eval.get_precision(),
                                                                coref_eval.get_f1()))
    muc, b_cubed, ceafe, avg = conll_eval.get_metrics()
    results['mlm'] = masked_lm_acc
    results['pairwise'] = (coref_eval.get_precision().item(), 
                           coref_eval.get_recall().item(), 
                           coref_eval.get_f1().item())
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

         
def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', type=str, default='configs/config_video_m2e2_mlm.json')
  parser.add_argument('--evaluate_only', action='store_true')
  parser.add_argument('--compute_confidence_bound', action='store_true')
  args = parser.parse_args()
  
  config = pyhocon.ConfigFactory.parse_file(args.config)
  print(config['model_path'])
  if not os.path.isdir(config['model_path']):
    os.makedirs(config['model_path'])
  if not os.path.isdir(os.path.join(config['model_path'], 'log')):
    os.mkdir(os.path.join(config['model_path'], 'log'))
  
  # Set up logger
  logging.basicConfig(filename=os.path.join(config['model_path'], 'log/{}.txt'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))),
                      format='%{asctime}s %(message)s', datefmt='%Y-%m_%d %H:%M:%S', level=logging.INFO)

  # Initialize dataloaders
  splits = [os.path.join(config['data_folder'], 'train_events.json'),\
            os.path.join(config['data_folder'], 'test_events.json')]
  feature_stoi = create_feature_stoi(splits, feature_types=config['linguistic_feature_types'])
  if args.compute_confidence_bound:
    seeds = [1111, 2222, 3333, 4444]
  else:
    seeds = [1111]

  for seed in seeds:
    config.random_seed = seed
    config['random_seed'] = seed
    fix_seed(config)

    train_set = TextEventMLMDataset(feature_stoi, 'train')
    test_set = TextEventMLMDataset(feature_stoi, 'test')
    
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True)
    
    event_model = AutoModel.from_pretrained(config.bert_model)

    train(event_model, 
          train_loader, 
          test_loader, 
          args, 
          random_seed=seed)
   

if __name__ == '__main__':
  main()
   
