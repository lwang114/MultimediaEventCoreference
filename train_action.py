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
from text_models import NoOp, BiLSTM
from corpus_action import VideoM2E2ActionDataset, fix_embedding_length

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


def train(visual_model, classifier, train_loader, test_loader, args): # TODO
  config = pyhocon.ConfigFactory.parse_file(args.config)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_grad_enabled(True)

  visual_model.to(device)
  classifier.to(device)

  # Create exp
  if not os.path.isdir(args.exp_dir):
    os.path.mkdir(args.exp_dir)
    
  # Define the training criterion
  criterion = nn.CrossEntropyLoss()
  
  # Set up the optimizer
  optimizer = get_optimizer(config, [text_model, visual_model])

  # Start training
  total_loss = 0.
  total = 0.
  best_f1 = 0
  begin_time = time.time()
  if args.evaluate_only:
    config.epochs = 0

  for epoch in range(config.epochs):
    visual_model.train()
    classifier.train()
    for i, batch in enumerate(train_loader):
      action_embeddings,\
      action_masks,\
      labels = batch

      B = action_embeddings.size(0)
      action_embeddings = action_embeddings.to(device)
      action_masks = action_masks.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      
      action_outputs = visual_model(action_embeddings).sum(dim=1) / action_masks.sum(-1).unsqueeze(-1)
      scores = classifier(action_outputs)
      loss = criterion(scores, labels)
      loss.backward()
      optimizer.step()

      total_loss += loss.item() * B
      total += B
      if i % 100 == 0:
        info = 'Iter {} {:.4f}'.format(i, total_loss / total)
        print(info)
        logging.info(info)

    if epoch % 5 == 0:
      macro_f1 = test(visual_model, test_loader, args)
      if macro_f1 > best_f1:
        best_f1 = macro_f1    
        torch.save(visual_model.module.state_dict(), '{}/visual_model.pth'.format(args.exp_dir))
        torch.save(visual_model.module.state_dict(), '{}/classifier.pth'.format(args.exp_dir))
    info = 'Epoch: [{}][{}/{}]\tTime {:.3f}\tLoss total {:.4f} ({:.4f})\tBest F1 {:.3f}'.format(epoch, i, len(train_loader), time.time()-begin_time, total_loss, total_loss / total, best_f1)
    print(info)
    logging.info(info)

  if args.evaluate_only:
    test(visual_model, train_loader, args)
    test(text_model, train_loader, args)

def test(visual_model, classifier, test_loader, args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  config = pyhocon.ConfigFactor.parse_file(args.config)
  v_feat_type = config["visual_feature_type"]
  all_scores = []
  all_labels = []

  visual_model.eval()
  classifier.eval(0
  with torch.no_grad():
    pred_dicts = []
    embs = dict()
    action_class_labels = dict()
    f_out = open(os.path.join(config['model_path'], 'prediction.readable'), 'w') 
    f_out.write('Document id\taction info\tscore\tlabel\n')
    for i, batch in enumerate(test_loader):
      action_embedding,\
      action_mask,\
      label = batch

      action_embedding = action_embedding.to(device)
      action_mask = action_mask.to(device)
      action_output = visual_model(action_embedding).sum(dim=1) / action_mask.sum(-1).unsqueeze(-1)
      scores = classifier(action_output)
      all_scores.append(torch.softmax(scores, axis=-1))
      all_labels.append(label)

      for idx in range(action_embedding.size(0)):
        global_idx = i * test_loader.batch_size + idx
        doc_id, span, label = test_loader.dataset.data_list[global_idx]
        if not doc_id in embs:
          embs[doc_id] = dict()
          action_class_labels[doc_id] = []

        embs[doc_id] = action_output[idx].detach().cpu().numpy()
        action_class_labels[doc_id].append(label)

        f_out.write(f'{doc_id}\t({span})\t{scores[idx]}\t{label[idx]}\n')
    embs = {f'{doc_id}_{doc_idx}':np.stack([embs[doc_id][span] for span in sorted(embs[doc_id])]) for doc_idx, doc_id in enumerate(sorted(embs))}
    np.savez(os.path.join(config['model_path'], f'{test_loader.dataset.split}_{v_feat_type}_finetuned.npz'), **embs)
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)
    
    preds = all_scores.max(axis=-1)[1]
    acc = sklearn.metrics.f1_score(all_labels.detach().cpu().numpy(),
                                   preds.detach().cpu().numpy(),
                                   average='micro')
    macro_f1 = sklearn.metrics.f1_score(all_labels.detach().cpu().numpy(),
                                   preds.detach().cpu().numpy(),
                                   average='macro')
    weighted_f1 = sklearn.metrics.f1_score(all_labels.detach().cpu().numpy(),
                                   preds.detach().cpu().numpy(),
                                   average='weighted')
 
    print(f'Accuracy: {acc:.3f}\tMacro F1: {macro_f1:.3f}\tWeighted Macro F1: {weighted_f1:.3f}')
    return macro_f1

if __name__ == '__main__':
  # Set up argument parser
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', type=str, default='')
  parser.add_argument('--config', type=str, default='config/config_action_video_m2e2.json')
  parser.add_argument('--evaluate_only', action='store_true')
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda_is_available() else 'cpu')
  config = pyhocon.ConfigFactory.parse_file(args.config)
  if not args.exp_dir:
    args.exp_dir = config['model_path']
  else:
    config['model_path'] = args.exp_dir

  if not os.path.isdir(config['log_path']):
    os.mkdir(config['log_path']) 
  logging.basicConfig(filename=os.path.join(config['log_path'],'{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                      format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) 

  # Initialize dataloaders
  train_set = VideoM2E2ActionDataset(config, split='train')
  test_set = VideoM2E2ActionDataset(config, split='test')
  fix_seed(config)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

  # Initialize models
  visual_model = BiLSTM(400, 400, num_layers=3)
  classifier = nn.Linear(800, len(train_set.ontology))

  if config['training_method'] == 'continue':
    visual_model.load_state_dict(torch.load(config['visual_model_path']))
    classifier.load_state_dict(torch.load(config['classifier_path']))
  
  # Training
  train(visual_model, classifier, train_loader, test_loader, args)
