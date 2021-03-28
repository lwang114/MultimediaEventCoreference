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
from image_models import VisualEncoder
from criterion import TripletLoss
from corpus import SupervisedGroundingFeatureDataset
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

def train(text_model, mention_model, image_model, coref_model, train_loader, test_loader, args, random_seed=None):
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
  
  if not isinstance(coref_model, torch.nn.DataParallel):
    coref_model = nn.DataParallel(coref_model)

  text_model.to(device)
  mention_model.to(device)
  image_model.to(device)
  coref_model.to(device)

  # Create/load exp
  if args.start_epoch != 0:
    text_model.load_state_dict(torch.load('{}/text_model.pth'.format(config['model_path'], args.start_epoch)))
    image_model.load_state_dict(torch.load('{}/image_model.pth'.format(config['model_path'], args.start_epoch)))
    coref_model.load_state_dict(torch.load('{}/text_scorer.{}.pth'.format(config['model_path'], args.start_epoch)))

  # Define the training criterion
  criterion = nn.BCEWithLogitsLoss()
  multimedia_criterion = TripletLoss(config) 

  # Set up the optimizer  
  optimizer = get_optimizer(config, [text_model, image_model, mention_model, coref_model])
   
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
      doc_embeddings, start_mappings, end_mappings, continuous_mappings,\
      width, videos,\
      text_labels, type_labels, img_labels,\
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
      type_labels = type_labels.to(device)
      text_mask = text_mask.to(device)
      span_mask = span_mask.to(device)
      video_mask = video_mask.to(device)
      first_grounding_idx, second_grounding_idx, pairwise_grounding_labels = get_pairwise_labels(text_labels, img_labels, is_training=False, device=device)
      first_text_idx, second_text_idx, pairwise_text_labels = get_pairwise_text_labels(text_labels, is_training=False, device=device)      
      pairwise_grounding_labels = pairwise_grounding_labels.to(torch.float)
      pairwise_text_labels = pairwise_text_labels.to(torch.float).flatten()
      if first_grounding_idx is None or first_text_idx is None:
        continue
      optimizer.zero_grad()

      video_output = image_model(videos)
      text_output = text_model(doc_embeddings)
      # XXX combined_output = text_model(torch.cat([video_output, doc_embeddings], dim=1))
      # video_output2 = combined_output[:, :video_output.size(1)]
      # text_output = combined_output[:, video_output.size(1):]      
      mention_output = mention_model(text_output,
                                     start_mappings, end_mappings,
                                     continuous_mappings, width,
                                     type_labels=type_labels)

      scores = []
      for idx in range(B):
          scores.append(coref_model(mention_output[idx, first_text_idx[idx]],
                                    mention_output[idx, second_text_idx[idx]]))
      scores = torch.cat(scores).squeeze(1)
      loss = criterion(scores, pairwise_text_labels)
      # loss = loss + multimedia_criterion(doc_embeddings, video_output, text_mask, video_mask) # XXX
      loss = loss + multimedia_criterion(text_output, video_output, text_mask, video_mask)
      
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
    torch.save(image_model.module.state_dict(), '{}/image_model-{}.pth'.format(config['model_path'], config['random_seed']))
    torch.save(mention_model.module.state_dict(), '{}/mention_model-{}.pth'.format(config['model_path'], config['random_seed']))
    torch.save(coref_model.module.state_dict(), '{}/coref_model-{}.pth'.format(config['model_path'], config['random_seed']))
 
    if epoch % 1 == 0:
      task = config.get('task', 'coreference')
      if task in ('coreference', 'both'):
        res = test(text_model, mention_model, image_model, coref_model, test_loader, args)
        if res['pairwise'][-1] >= best_text_f1:
          best_text_f1 = res['pairwise'][-1]
          results['pairwise'] = res['pairwise']
          results['muc'] = res['muc']
          results['ceafe'] = res['ceafe']
          results['bcubed'] = res['bcubed']
          results['avg'] = res['avg']
          torch.save(text_model.module.state_dict(), '{}/best_text_model-{}.pth'.format(config['model_path'], config['random_seed']))
          torch.save(image_model.module.state_dict(), '{}/best_image_model-{}.pth'.format(config['model_path'], config['random_seed']))
          torch.save(mention_model.module.state_dict(), '{}/best_mention_model-{}.pth'.format(config['model_path'], config['random_seed']))
          torch.save(coref_model.module.state_dict(), '{}/best_coref_model-{}.pth'.format(config['model_path'], config['random_seed']))
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
      results = test(text_model, mention_model, image_model, coref_model, test_loader, args)
    if task in ('retrieval', 'both'):
      I2S_r10, S2I_r10 = test_retrieve(text_model, image_model, test_loader, args)
  return results
      
def test(text_model, mention_model, image_model, coref_model, test_loader, args): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = pyhocon.ConfigFactory.parse_file(args.config)
    documents = test_loader.dataset.documents
    all_scores = []
    all_labels = []

    text_model.eval()
    image_model.eval()
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
          width, videos,\
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
          videos = videos.to(device)
          text_labels = text_labels.to(device)
          type_labels = type_labels.to(device)
          img_labels = img_labels.to(device)
          text_mask = text_mask.to(device)
          span_mask = span_mask.to(device)
          video_mask = video_mask.to(device)

          # Extract span and video embeddings
          video_output = image_model(videos)
          text_output = text_model(doc_embeddings)
          # XXX combined_output = text_model(torch.cat([video_output, doc_embeddings], dim=1))
          # video_output = combined_output[:, :video_output.size(1)]
          # text_output = combined_output[:, video_output.size(1):]
          mention_output = mention_model(text_output,
                                         start_mappings, end_mappings,
                                         continuous_mappings, width,
                                         type_labels=type_labels)
          
          # Compute score for each mention pair
          B = doc_embeddings.size(0) 
          for idx in range(B):
            global_idx = i * test_loader.batch_size + idx

            # Compute pairwise labels
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
            # Save the output clusters
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

        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))

        print('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        print('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                     len(all_labels)))
        print('Pairwise - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                eval.get_precision(), eval.get_f1()))
        
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
    with open(os.path.join(config['model_path'], 'I2S.txt'), 'w') as f_i2s,\
         open(os.path.join(config['model_path'], 'S2I.txt'), 'w') as f_s2i:
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
  parser.add_argument('--config', type=str, default='configs/config_coref_simple_video_m2e2.json')
  parser.add_argument('--start_epoch', type=int, default=0)
  parser.add_argument('--evaluate_only', action='store_true')
  parser.add_argument('--compute_confidence_bound', action='store_true')
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # Set up logger
  config = pyhocon.ConfigFactory.parse_file(args.config)

  print(config['model_path'])
  if not os.path.isdir(config['model_path']):
      os.makedirs(config['model_path'])
  if not os.path.isdir(os.path.join(config['model_path'], 'log')):
      os.mkdir(os.path.join(config['model_path'], 'log')) 

  logging.basicConfig(filename=os.path.join(config['model_path'],'log/{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),\
                      format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO) 

  train_set = SupervisedGroundingFeatureDataset(os.path.join(config['data_folder'], 'train.json'), 
                                                os.path.join(config['data_folder'], f'train_{config.mention_type}.json'), 
                                                os.path.join(config['data_folder'], 'train_bboxes.json'),
                                                config, split='train')
  test_set = SupervisedGroundingFeatureDataset(os.path.join(config['data_folder'], 'test.json'), 
                                               os.path.join(config['data_folder'], f'test_{config.mention_type}.json'), 
                                               os.path.join(config['data_folder'], 'test_bboxes.json'), 
                                               config, split='test')

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
      text_model = nn.TransformerEncoderLayer(d_model=config.hidden_layer, nhead=1)
      image_model = VisualEncoder(400, config.hidden_layer)

      mention_model = SpanEmbedder(config, device)
      if config['classifier'] == 'simple':
          coref_model = SimplePairWiseClassifier(config).to(device)
      elif config['classifier'] == 'attention':
          coref_model = TransformerPairWiseClassifier(config).to(device)
      else:
          raise ValueError(f"Invalid classifier type {config['classifier']}")
      
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
      results = train(text_model, mention_model, image_model, coref_model, train_loader, test_loader, args, random_seed=seed)
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
