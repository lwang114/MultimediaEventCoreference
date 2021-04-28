import nltk
import numpy as np
import os
import json
import logging
import torch
from itertools import combinations
import argparse
from evaluator import Evaluation
from conll import write_output_file

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)
format_str = '%(asctime)s\t%(message)s'
console.setFormatter(logging.Formatter(format_str))

NEW = '###NEW###'
EPS = 1e-40
class SMTCoreferencer:
  def __init__(self, doc_path, mention_path, embed_path, out_path, config):
    '''
    Unsupervised coreference system based on
    ``Unsupervised Ranking Model for Entity Coreference Resolution``, 
    X. Ma, Z. Liu and E. Hovy, NAACL, 2016.
    
    :param doc_path: path to the mapping of
            [doc_id]: list of [sent id, token id, token, is entity/event]
    :param mention_path: path to the list of dicts of:
            {'doc_id': str,
             'sentence_id': str,
             'tokens_ids': list of ints,
             'cluster_id': '0',
             'tokens': str, tokens concatenated with space} 
    '''
    self.doc_path = doc_path
    self.mention_path = mention_path
    self.embed_path = embed_path
    self.out_path = out_path
    self.max_mention_length = config.get('max_mention_length', 20)
    self.max_num_mentions = config.get('max_num_mentions', 200)
    self.is_one_indexed = config.get('is_one_indexed', False)
    if not os.path.exists(self.out_path):
      os.makedirs(self.out_path)
    
    self.mentions,\
    self.postags,\
    self.embeddings,\
    self.tokens_ids,\
    self.cluster_ids,\
    self.vocabs = self.load_corpus(doc_path, mention_path, embed_path)
    self.doc_ids = sorted(json.load(open(doc_path)))
    logger.info('Number of documents = {}'.format(len(self.mentions)))
    logger.info('Vocab size = {}'.format(len(self.vocabs)))

    self.mention_probs = 1. / len(self.vocabs) * np.ones((2, len(self.vocabs), len(self.vocabs)))
    self.alpha0 = EPS

  def load_corpus(self, doc_path, mention_path, embed_path):
    '''
    :returns doc_path: str,
    :returns mention_path: str    
    '''
    documents = json.load(open(doc_path))
    mention_list = json.load(open(mention_path))
    token_embeddings = np.load(embed_path)

    mention_dict = {}
    mentions = []
    spans = []
    postags = []
    cluster_ids = []
    embeddings = []
    doc_ids = sorted(documents)
    feat_keys_all = sorted(token_embeddings, key=lambda x:int(x.split('_')[-1]))
    feat_keys = []
    for doc_id in doc_ids:
      for feat_key in feat_keys_all:
        if '_'.join(feat_key.split('_')[:-1]) == doc_id:
          feat_keys.append(feat_key)
    
    embed_dim = token_embeddings[feat_keys[0]].shape[-1]
    vocabs = {'###UNK###': 0}
    
    for m in mention_list:
      # Create a mapping from span to cluster id
      tokens_ids = m['tokens_ids']
      span = (min(tokens_ids)-self.is_one_indexed, max(tokens_ids)-self.is_one_indexed)
      cluster_id = m['cluster_id']
      if not m['doc_id'] in mention_dict:
        mention_dict[m['doc_id']] = {}
      mention_dict[m['doc_id']][span] = cluster_id

    for doc_id, feat_id in zip(sorted(documents)[:20], feat_keys): # XXX
      if not doc_id in mention_dict:
        continue
      mentions.append([[NEW]])
      postags.append([[NEW]])
      spans.append([[-1, -1]])
      cluster_ids.append([-1])
      embeddings.append([np.random.normal(size=(1, embed_dim))])

      tokens = [t[2] for t in documents[doc_id]]
      postag = [t[1] for t in nltk.pos_tag(tokens)]

      for span in sorted(mention_dict[doc_id])[:self.max_num_mentions]:
        cluster_id = mention_dict[doc_id][span]
        mention_token = tokens[span[0]:span[1]+1][:self.max_mention_length]
        mention_tag = postag[span[0]:span[1]+1][:self.max_mention_length]
        mention_emb = token_embeddings[feat_id][span[0]:span[1]+1][:self.max_mention_length]

        if not ' '.join(mention_token) in vocabs:
          vocabs[' '.join(mention_token)] = len(vocabs)

        mentions[-1].append(mention_token)
        postags[-1].append(mention_tag)
        embeddings[-1].append(mention_emb)
        spans[-1].append(span)
        cluster_ids[-1].append(cluster_id)

    vocabs[NEW] = len(vocabs)
    return mentions, postags, embeddings, spans, cluster_ids, vocabs

  def is_match(self, first, second):
    first = [f.lower() for f in first]
    second = [s.lower() for s in second]
    return ' '.join(first) == ' '.join(second)
    
  def is_compatible(self, first_tags, second_tags):
    for first_tag in first_tags:
      for second_tag in second_tags:
        if first_tag[:2] == second_tag[:2]:
          return True
        elif first_tag[:2] in ['NN', 'VB', 'PR'] and second_tag[0] in ['NN', 'VB', 'PR']:
          return True
        elif first_tag == NEW:
          return True
    return False
  
  def set_mode(self, mention, antecedents):
    for ant in antecedents:
      if self.is_match(ant, mention):
        return 0
    return 1

  def similarity(self, first_embs, second_embs, first_tokens, second_tokens):
    if NEW in first_tokens:
      return 1. 
    first_embs /= (np.linalg.norm(first_embs, ord=2, axis=-1, keepdims=True) + EPS)
    second_embs /= (np.linalg.norm(second_embs, ord=2, axis=-1, keepdims=True) + EPS)
    sim_map = np.abs(first_embs @ second_embs.T)
    return sim_map.max(axis=0).mean()

  def fit(self, n_epochs=10):
    best_f1 = 0.
    for epoch in range(n_epochs):
      N_c = [np.zeros((2, len(sent), len(sent))) for sent in self.mentions]
      N_m = np.zeros((2, len(self.vocabs), len(self.vocabs)))

      for sent_idx, (sent, postag, embedding) in enumerate(zip(self.mentions, self.postags, self.embeddings)):
        for second_idx in range(min(len(sent), self.max_num_mentions)):
          mode = self.set_mode(sent[second_idx], sent[:second_idx])
          for first_idx in range(second_idx):
            if mode == 0 and self.is_match(sent[first_idx], sent[second_idx]):
              N_c[sent_idx][0, first_idx, second_idx] += 1
            elif mode == 1 and self.is_compatible(postag[first_idx], postag[second_idx]):
              first_token_idx = self.vocabs[' '.join(sent[first_idx])]
              second_token_idx = self.vocabs[' '.join(sent[second_idx])] 
              N_c[sent_idx][1, first_idx, second_idx] += self.mention_probs[1, first_token_idx, second_token_idx] * self.similarity(embedding[first_idx], embedding[second_idx], sent[first_idx], sent[second_idx])
        N_c[sent_idx] /= N_c[sent_idx].sum(axis=1, keepdims=True) + EPS
         
        for second_idx in range(min(len(sent), self.max_num_mentions)):
          for first_idx in range(second_idx):
            for mode in range(2):
              first_token_idx = self.vocabs[' '.join(sent[first_idx])]
              second_token_idx = self.vocabs[' '.join(sent[second_idx])] 
              N_m[mode, first_token_idx, second_token_idx] += N_c[sent_idx][mode, first_idx, second_idx]

      # Update mention probs
      self.mention_probs = N_m / (N_m.sum(axis=-1, keepdims=True) + EPS)

      logger.info('Epoch {}, log likelihood = {:.3f}'.format(epoch, self.log_likelihood()))
      # Compute pairwise coreference scores
      pairwise_f1 = self.evaluate(self.doc_path,
                                  self.mention_path,
                                  self.embed_path,
                                  self.out_path)

      f = open(os.path.join(self.out_path, 'alignment_counts.readable'), 'w')
      print(os.path.join(self.out_path, 'alignment_counts.readable')) # XXX
      for sent_idx, (sent, postag) in enumerate(zip(self.mentions, self.postags)):
        second_idxs = [i for i in range(min(len(sent), self.max_num_mentions)) for _ in range(i)]
        first_idxs = [j for i in range(min(len(sent), self.max_num_mentions)) for j in range(i)]
        N_c_max = [max(N_c[sent_idx][:, i, j]) for i, j in zip(first_idxs, second_idxs)]
        top_pairs = np.argsort(-np.asarray(N_c_max))[:100]
        for pair_idx in top_pairs:
          first_idx = first_idxs[pair_idx]
          second_idx = second_idxs[pair_idx]
          f.write('{} {} {} {} {}: {}\n'.format(self.doc_ids[sent_idx], first_idx, second_idx, ' '.join(sent[first_idx]), ' '.join(sent[second_idx]), N_c_max[pair_idx]))
        
      np.save(os.path.join(self.out_path, '../translate_probs.npy'), self.mention_probs)
      if pairwise_f1 > best_f1:
        best_f1 = pairwise_f1
        np.save(os.path.join(self.out_path, '../translate_probs_best.npy'), self.mention_probs)
        vocab_list = sorted(self.vocabs, key=lambda x:self.vocabs[x])
        with open(os.path.join(self.out_path, '../translate_probs_top200.txt'), 'w') as f:
          top_idxs = np.argsort(-N_m.sum(axis=(0, -1)))[:200]
          for v_idx in top_idxs:
            v = vocab_list[v_idx]
            for mode in range(2):
              p = self.mention_probs[mode, v_idx] 
              top_coref_idxs = np.argsort(-p)[:100]
              for c_idx in top_coref_idxs:
                f.write('Mode {}: {} -> {}: {}\n'.format(mode, v, vocab_list[c_idx], p[c_idx]))
              f.write('\n')
 
  def log_likelihood(self):
    ll = -np.inf
    N = len(self.mentions)
    for sent_idx, (mentions, postags, embeddings) in enumerate(zip(self.mentions, self.postags, self.embeddings)):
      if sent_idx == 0:
        ll = 0
      
      for m_idx, (second, second_tag, second_emb) in enumerate(zip(mentions, postags, embeddings)):
        if m_idx == 0:
          continue
        second_token_idx = self.vocabs[' '.join(second)]
        mode = self.set_mode(second, mentions[:m_idx])
        second_tag = postags[m_idx]
        
        p_sent = 0.
        for m_idx2, (first, first_tag, first_emb) in enumerate(zip(mentions[:m_idx], postags[:m_idx], embeddings[:m_idx])):
          first_token_idx = self.vocabs[' '.join(first)]
          p_sent += 1. / m_idx * self.mention_probs[mode, first_token_idx, second_token_idx] *\
                    self.similarity(first_emb, second_emb, first, second)
        ll += 1. / N * np.log(p_sent + EPS)
    return ll

  def predict(self, mentions, spans, postags, embeddings):
    '''
    :param mentions: a list of list of str,
    :param spans: a list of tuple of (start idx, end idx)
    :returns clusters:
    ''' 
    clusters = {}
    cluster_labels = {}

    for second_idx in range(1, len(mentions)):
      scores = []
      mode = self.set_mode(mentions[second_idx], mentions[:second_idx])
  
      second = mentions[second_idx]
      second_emb = embeddings[second_idx]
      for first_idx in range(second_idx):
        first = mentions[first_idx]
        if mode == 0 and self.is_match(mentions[first_idx], mentions[second_idx]):
          scores.append(1.)
        elif mode == 1 and self.is_compatible(postags[first_idx], postags[second_idx]):
          first_token_idx = self.vocabs.get(' '.join(mentions[first_idx]), 0)
          second_token_idx = self.vocabs.get(' '.join(mentions[second_idx]), 0)
          first_emb = embeddings[first_idx]
          scores.append(self.mention_probs[1, first_token_idx, second_token_idx]\
                        * self.similarity(first_emb, second_emb, first, second))
        else:
          scores.append(0.)
      scores = np.asarray(scores)
      
      a_idx = np.argmax(scores)
      if a_idx == 0:
        c_idx = len(clusters)
        clusters[c_idx] = [second_idx-1]
        cluster_labels[second_idx-1] = c_idx 
      else:      
        c_idx = cluster_labels[a_idx-1]
        cluster_labels[second_idx-1] = c_idx
        clusters[c_idx].append(second_idx-1)
    return clusters, cluster_labels

  def evaluate(self, doc_path, mention_path, embed_path, out_path):
    if not os.path.exists(out_path):
      os.makedirs(out_path)

    documents = json.load(open(doc_path, 'r'))
    doc_ids = sorted(documents)
    mentions_all, postags_all, embeddings_all, spans_all, cluster_ids_all, _ = self.load_corpus(doc_path, mention_path, embed_path)
    preds = []
    golds = []
    for doc_id, postags, mentions, embeddings, spans, cluster_ids in zip(doc_ids, postags_all, mentions_all, embeddings_all, spans_all, cluster_ids_all):
      clusters, cluster_labels = self.predict(mentions, spans, postags, embeddings)
      pred = [cluster_labels[f] == cluster_labels[s] for f, s in combinations(range(len(spans)-1), 2)]
      pred = np.asarray(pred)

      gold = [cluster_ids[f] != 0 and cluster_ids[s] != 0 and cluster_ids[f] == cluster_ids[s] for f, s in combinations(range(len(spans)-1), 2)]
      gold = np.asarray(gold)

      preds.extend(pred)
      golds.extend(gold)

      # Save output files to CoNLL format
      document = {doc_id:documents[doc_id]}
      spans = np.asarray(spans)
      write_output_file(document, clusters,
                        [doc_id]*(len(spans)-1),
                        spans[1:, 0].tolist(),
                        spans[1:, 1].tolist(),
                        out_path,
                        doc_id)
    
    preds = torch.from_numpy(np.asarray(preds))
    golds = torch.from_numpy(np.asarray(golds))
    pairwise_eval = Evaluation(preds, golds) 
    
    precision = pairwise_eval.get_precision()
    recall = pairwise_eval.get_recall()
    f1 = pairwise_eval.get_f1()
    logger.info('Pairwise Recall = {:.3f}, Precision = {:.3f}, F1 = {:.3f}'.format(recall, precision, f1))
    return f1 

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--dataset', choices={'ecb', 'video_m2e2', 'ace'}, default='ecb')
  parser.add_argument('--mention_type', choices={'events', 'entities', 'mixed'}, default='mixed')
  parser.add_argument('--exp_dir', default=None)
  args = parser.parse_args()
  
  dataset = args.dataset
  coref_type = args.mention_type
  doc_path_train = 'data/{}/mentions/train.json'.format(dataset)
  mention_path_train = 'data/{}/mentions/train_{}.json'.format(dataset, coref_type)
  embed_path_train = 'data/{}/mentions/train_glove_embeddings.npz'.format(dataset)
  doc_path_test = 'data/{}/mentions/test.json'.format(dataset)
  mention_path_test = 'data/{}/mentions/test_{}.json'.format(dataset, coref_type)
  embed_path_test = 'data/{}/mentions/test_glove_embeddings.npz'.format(dataset)
  out_path = args.exp_dir
  if not args.exp_dir:
    out_path = 'models/smt_{}/pred_{}'.format(dataset, coref_type)
    
  model = SMTCoreferencer(doc_path_train, mention_path_train, embed_path_train,
                          out_path=os.path.join(out_path, '{}_train'.format(args.mention_type)),
                          config={'is_one_indexed': dataset == 'ecb'})
  model.fit(2)
  model.evaluate(doc_path_test, mention_path_test, embed_path_test,
                 out_path=os.path.join(out_path, '{}_test'.format(args.mention_type)))
