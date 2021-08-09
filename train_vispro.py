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
from itertools import combinations
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from text_models import BiLSTM, SpanEmbedder, SimplePairWiseClassifier
from visual_models import BiLSTMVideoEncoder, CrossmediaPairWiseClassifier, ClassAttender, VisCorefAttender
from corpus import TextVideoEventDataset
from vispro import VisProDataset
from evaluator import Evaluation, CoNLLEvaluation
from utils import create_type_stoi, create_feature_stoi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NULL = '###NULL###'
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

def plot_attention(attention,
                   spans,
                   tokens,
                   class_labels,
                   class_names,
                   out_file):
    fig, ax = plt.subplots(figsize=(7, 10))
    plt.pcolor(attention.T, vmin=0., vmax=1., cmap=plt.cmap.Blues)
    span_strs = [tokens[span[0]:span[1]+1] for span in spans]
    class_strs = [class_names[c] for c in class_labels]
    num_spans = len(spans)
    num_classes = len(class_labels)
    for i in range(num_spans):
      for j in range(num_classes):
        plt.text(i, j, round(attention[i, j], 2), color='orange')

    plt.set_xticks([i+0.5 for i in range(num_spans)])
    plt.set_yticks([j+0.5 for j in range(num_classes)])
    plt.set_xticklabels(span_strs)
    plt.set_yticklabels(class_strs)
    plt.xticks(rotation=45, fontsize=30)
    plt.colorbar()
    plt.savefig(out_file)
    plt.close()

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
  n_class = len(train_loader.dataset.class_stoi)
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

  if not isinstance(text_coref_model, torch.nn.DataParallel):
    text_coref_model = nn.DataParallel(text_coref_model)

  if not isinstance(visual_coref_model, torch.nn.DataParallel):
    visual_coref_model = nn.DataParallel(visual_coref_model)

  text_model.to(device)
  mention_model.to(device)
  visual_model.to(device)
  attention_model.to(device)
  text_coref_model.to(device)
  visual_coref_model.to(device)

  # Define the training criterion
  criterion = nn.BCEWithLogitsLoss()
  optimizer = get_optimizer(config, [text_model,
                                     mention_model, 
                                     visual_model,
                                     attention_model, 
                                     text_coref_model,
                                     visual_coref_model])
 
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
    mention_model.train()
    visual_model.train()
    attention_model.train()
    text_coref_model.train()
    visual_coref_model.train()

    for i, batch in enumerate(train_loader):
      doc_embeddings = batch['doc_embeddings'].to(device)
      start_mappings = batch['start_mappings'].to(device)
      end_mappings = batch['end_mappings'].to(device)
      continuous_mappings = batch['continuous_mappings'].to(device)
      width = batch['width'].to(device)

      # videos = batch['visual_embeddings'].to(device)
      class_labels = batch['class_labels'].to(device)
      type_labels = batch['type_labels'].to(device)
      pairwise_labels = batch['pairwise_labels'].to(device)
      first_idxs = batch['first_idxs'].to(device) 
      second_idxs = batch['second_idxs'].to(device)

      text_mask = batch['text_mask'].to(device)
      span_mask = batch['span_mask'].to(device)
      class_mask = batch['class_mask'].to(device)
      mention_mask = (span_mask.sum(-1) > 0).float()

      pair_num = (first_idxs > 0).long().sum(-1)
      class_num = (class_mask.sum(-1) > 0).long().sum(-1)

      binary_class_labels = F.one_hot(class_labels, n_class).float() * class_mask.unsqueeze(-1)
      
      # Compute mention embeddings
      if 'type' in config.get('linguistic_feature_types', []):
          linguistic_labels = type_labels.unsqueeze(-1)
      else:
          linguistic_labels = None
          
      mention_embedding = mention_model(doc_embeddings,
                                        start_mappings,
                                        end_mappings,
                                        continuous_mappings, 
                                        width, 
                                        attention_mask=text_mask,
                                        linguistic_labels=linguistic_labels)
      
      # Compute textual event logits of size (batch size, span num, n event class)
      mention_output = text_model(mention_embedding)
      attn_weights = attention_model(mention_output, 
                                     binary_class_labels, 
                                     mention_mask, 
                                     class_mask)
  
      # Compute coreference scores
      text_scores = []
      pairwise_text_labels = []
      B = doc_embeddings.size(0)
      for idx in range(B):
        first_text_idx = first_idxs[idx, :pair_num[idx]].cpu().detach().numpy()
        second_text_idx = second_idxs[idx, :pair_num[idx]].cpu().detach().numpy()

        if not len(first_text_idx):
          continue
        pairwise_text_label = pairwise_labels[idx, :pair_num[idx]]
        visual_score = visual_coref_model.module.crossmedia_score(first_text_idx,
                                                                  second_text_idx,
                                                                  attn_weights[idx],
                                                                  use_null=True)

        text_score = text_coref_model(mention_embedding[idx, first_text_idx],
                                      mention_embedding[idx, second_text_idx])
        text_score = weight_visual * visual_score + (1 - weight_visual) * text_score
        text_scores.append(text_score)
        pairwise_text_labels.append(pairwise_text_label)
        
      if not len(text_scores):
        continue

      text_scores = torch.cat(text_scores).squeeze(1)
      pairwise_text_labels = torch.cat(pairwise_text_labels).to(torch.float)
      loss = criterion(text_scores, pairwise_text_labels)
        
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
    torch.save(text_coref_model.module.state_dict(), '{}/text_coref_model-{}.pth'.format(config['model_path'], random_seed))
    torch.save(visual_coref_model.module.state_dict(), '{}/visual_coref_model-{}.pth'.format(config['model_path'], random_seed))
    
    if epoch % 1 == 0:
      res = test(text_model,
                 mention_model,
                 visual_model,
                 attention_model,
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
        torch.save(text_model.module.state_dict(), '{}/best_text_model-{}.pth'.format(config['model_path'], random_seed))
        torch.save(mention_model.module.state_dict(), '{}/best_mention_model-{}.pth'.format(config['model_path'], random_seed))
        torch.save(visual_model.module.state_dict(), '{}/best_visual_model-{}.pth'.format(config['model_path'], random_seed))
        torch.save(attention_model.module.state_dict(), '{}/best_attention_model-{}.pth'.format(config['model_path'], random_seed))
        torch.save(text_coref_model.module.state_dict(), '{}/best_text_coref_model-{}.pth'.format(config['model_path'], random_seed))
        torch.save(visual_coref_model.module.state_dict(), '{}/best_visual_coref_model-{}.pth'.format(config['model_path'], random_seed))
        print('Best text coreference F1={}'.format(best_text_f1))

  if args.evaluate_only:
    results = test(text_model,
                   mention_model,
                   visual_model,
                   attention_model,
                   text_coref_model,
                   visual_coref_model,
                   test_loader,
                   args)
  return results

def test(text_model,
         mention_model,
         visual_model,
         attention_model,
         text_coref_model,
         visual_coref_model,
         test_loader,
         args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = pyhocon.ConfigFactory.parse_file(args.config)
    weight_visual = config.weight_visual
    n_class = len(test_loader.dataset.class_stoi) # TODO Change name for video m2e2
    class_names = sorted(test_loader.dataset.class_stoi, key=lambda x:test_loader.dataset.class_stoi[x])
    
    all_scores = []
    all_labels = []
    
    text_model.eval()
    mention_model.eval()
    visual_model.eval()
    attention_model.eval()
    text_coref_model.eval()
    visual_coref_model.eval()

    conll_eval = CoNLLEvaluation()
    f_out = open(os.path.join(config['model_path'], 'prediction.readable'), 'w')
    best_f1 = 0.
    results = dict()
    with torch.no_grad():
      for i, batch in enumerate(test_loader):
        doc_embeddings = batch['doc_embeddings'].to(device)
        start_mappings = batch['start_mappings'].to(device)
        end_mappings = batch['end_mappings'].to(device)
        continuous_mappings = batch['continuous_mappings'].to(device)
        width = batch['width'].to(device)
        
        text_labels = batch['cluster_labels'].to(device) 
        class_labels = batch['class_labels'].to(device)
        type_labels = batch['type_labels'].to(device)
        pairwise_labels = batch['pairwise_labels'].to(device) 
        first_idxs = batch['first_idxs'].to(device)
        second_idxs = batch['second_idxs'].to(device)

        text_mask = batch['text_mask'].to(device)
        span_mask = batch['span_mask'].to(device)
        class_mask = batch['class_mask'].to(device)
        span_num = (span_mask.sum(-1) > 0).long().sum(-1)

        pair_num = (first_idxs > 0).long().sum(-1)
        class_num = (class_mask.sum(-1) > 0).long().sum(-1)
        mention_mask = (span_mask.sum(-1) > 0).float()

        binary_class_labels = F.one_hot(class_labels, n_class).float() * class_mask.unsqueeze(-1)
  
        # Compute mention embeddings
        if 'type' in config.get('linguistic_feature_types', []):
            linguistic_labels = type_labels.unsqueeze(-1)
        else:
            linguistic_labels = None
            
        mention_embedding = mention_model(doc_embeddings,
                                          start_mappings,
                                          end_mappings,
                                          continuous_mappings, 
                                          width,
                                          attention_mask=text_mask,
                                          linguistic_labels=linguistic_labels)
        
        # Compute textual event logits of size (batch size, span num, n event class)
        mention_output = text_model(mention_embedding)
        attn_weights = attention_model(mention_output, 
                                       binary_class_labels,
                                       mention_mask,
                                       class_mask)
        
        # Compute coreference scores
        text_scores = []
        pairwise_text_labels = []
        B = doc_embeddings.size(0)
        for idx in range(B):
          global_idx = i * test_loader.batch_size + idx
          first_text_idx = first_idxs[idx, :pair_num[idx]].cpu().detach().numpy()
          second_text_idx = second_idxs[idx, :pair_num[idx]].cpu().detach().numpy()
          pairwise_text_label = pairwise_labels[idx, :pair_num[idx]]

          if not len(first_text_idx):
            continue

          visual_score = visual_coref_model.module.crossmedia_score(first_text_idx,
                                                                    second_text_idx,
                                                                    attn_weights[idx],
                                                                    use_null=True)

          text_score = text_coref_model(mention_embedding[idx, first_text_idx],
                                        mention_embedding[idx, second_text_idx])
          text_score = weight_visual * visual_score + (1 - weight_visual) * text_score
          text_scores.append(text_score)
          pairwise_text_labels.append(pairwise_text_label)

          # Save readable predictions
          pred_str, gold_str = Evaluation.make_output_readable(first_text_idx,
                                                               second_text_idx,
                                                               (text_score > 0).long().cpu().detach().numpy(),
                                                               pairwise_text_label.cpu().detach().numpy(),
                                                               candidate_start_ends,
                                                               tokens)
          f_out.write(f"{doc_id}: {token_str}\n")
          f_out.write(f"{pred_str}\n")
          f_out.write(f"{gold_str}\n")

          # Save crossmedia attention maps
          attention = torch.softmax(attn_weights[idx, :span_num[idx], :class_num[idx]], dim=-1)
          plot_attention(attention.cpu().detach().numpy(), 
                         tokens, 
                         candidate_start_ends,
                         tokens,
                         class_labels,
                         class_names,
                         os.path.join(attention_path, doc_id+'.png'))

          all_scores.append(text_score.squeeze(1))
          all_labels.append(pairwise_text_label)
      f_out.close()
      all_scores = torch.cat(all_scores)
      all_labels = torch.cat(all_labels)
      
      average_precision = average_precision_score(all_labels.cpu().detach().numpy(),
                                                  torch.sigmoid(all_scores).cpu().detach().numpy())
      
      strict_preds = (all_scores > 0).to(torch.int)
      eval = Evaluation(strict_preds, all_labels.to(device))
      print('[Text Coreference Result]')
      print('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
      print('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                       len(all_labels)))
      info = 'Pairwise - Recall: {}, Precision: {}, F1: {}, mAP: {}'.format(eval.get_recall(),
                                                                            eval.get_precision(), 
                                                                            eval.get_f1(),
                                                                            average_precision)
      print(info)
      logging.info(info)
      
      """ TODO
      muc, b_cubed, ceafe, avg = conll_eval.get_metrics()
      conll_metrics = muc+b_cubed+ceafe+avg
      info = 'MUC - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '\
             'Bcubed - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '\
             'CEAFe - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, '\
             'CoNLL - Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(*conll_metrics)
      print(info) 
      logging.info(info)

      results['pairwise'] = (eval.get_precision().item(), eval.get_recall().item(), eval.get_f1().item(), average_precision)
      results['muc'] = muc
      results['bcubed'] = b_cubed
      results['ceafe'] = ceafe
      results['avg'] = avg
      """
      results['pairwise'] = (eval.get_precision().item(), eval.get_recall().item(), eval.get_f1().item(), average_precision)
      results['muc'] = (0, 0, 0)
      results['bcubed'] = (0, 0, 0)
      results['ceafe'] = (0, 0, 0)
      results['avg'] = (0, 0, 0)

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
  ontology_map_file = os.path.join(config['data_folder'], '../ontology_map.json')
  if os.path.exists(ontology_map_file):
    t2v_map = json.load(open(ontology_map_file))
    class_stoi = {t:i+1 for i, t in enumerate(sorted(t2v_map))} 
    class_stoi[NULL] = 0

  train_set = VisProDataset(config, split='train', class_stoi=class_stoi)
  test_set = VisProDataset(config, split='test', class_stoi=class_stoi)
  
  pairwises  = []
  mucs = []
  bcubeds = []
  ceafes = []
  avgs = []
  
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
      attention_model = VisCorefAttender(config.hidden_layer,
                                         config.hidden_layer,
                                         len(train_set.class_stoi))
      visual_model = BiLSTMVideoEncoder(400, int(config.hidden_layer // 2))
      visual_coref_model = CrossmediaPairWiseClassifier(config).to(device)
      
      if config['training_method'] in ('pipeline', 'continue') or args.evaluate_only:
        text_model.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                           f'best_text_model-{seed}.pth'), map_location=device))
        mention_model.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                              f'best_mention_model-{seed}.pth')))
        visual_coref_model.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                                   f'best_visual_coref_model-{seed}.pth'), map_location=device))
        text_coref_model.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                                 f'best_text_coref_model-{seed}.pth'), map_location=device))
      
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
  info = f'Pairwise: precision {mean_pairwise[0]} +/- {std_pairwise[0]}, '\
         f'recall {mean_pairwise[1]} +/- {std_pairwise[1]}, '\
         f'f1 {mean_pairwise[2]} +/- {std_pairwise[2]}\n'\
         f'mAP {mean_pairwise[3]} +/- {std_pairwise[3]}\n'\
         f'MUC: precision {mean_muc[0]} +/- {std_muc[0]}, '\
         f'recall {mean_muc[1]} +/- {std_muc[1]}, '\
         f'f1 {mean_muc[2]} +/- {std_muc[2]}\n'\
         f'Bcubed: precision {mean_bcubed[0]} +/- {std_bcubed[0]}, '\
         f'recall {mean_bcubed[1]} +/- {std_bcubed[1]}, '\
         f'f1 {mean_bcubed[2]} +/- {std_bcubed[2]}\n'\
         f'CEAFe: precision {mean_ceafe[0]} +/- {std_ceafe[0]}, '\
         f'recall {mean_ceafe[1]} +/- {std_ceafe[1]}, '\
         f'f1 {mean_ceafe[2]} +/- {std_ceafe[2]}\n'\
         f'CoNLL: precision {mean_avg[0]} +/- {std_avg[0]}, '\
         f'recall {mean_avg[1]} +/- {std_avg[1]}, '\
         f'f1 {mean_avg[2]} +/- {std_avg[2]}'
  print(info)
  logging.info(info) 
