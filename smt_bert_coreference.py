import nltk
import numpy as np
import os
import json
import logging
import argparse
import torch
from itertools import combinations
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
from transformers import AutoTokenizer
from evaluator import Evaluation
from conll import write_output_file

EPS = 1e-40
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)
format_str = '%(asctime)s\t%(message)s'
console.setFormatter(logging.Formatter(format_str))

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
    self.max_mention_length = config.get('max_mention_length', 20) # Maximum length of a mention
    self.max_num_mentions = config.get('max_num_mentions', 200)
    self.is_one_indexed = config.get('is_one_indexed', False)
    if not os.path.exists(self.out_path):
      os.makedirs(self.out_path)
    self.dep_parser = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz')
    self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    
    self.mentions,\
    self.postags,\
    self.embeddings,\
    self.tokens_ids,\
    self.cluster_ids,\
    self.vocabs = self.load_corpus(doc_path, mention_path, embed_path)
    self.mention_probs = 1. / len(self.vocabs) * np.ones((len(self.vocabs), len(self.vocabs)))
    self.length_probs = 1. / self.max_mention_length * np.ones((self.max_mention_length, self.max_mention_length))
    self.coref_probs = 1. / self.max_num_mentions * np.ones((self.max_num_mentions, self.max_num_mentions))
    self.alpha0 = EPS

  def tokenize(self, documents):
    '''
    Tokenize the sentences in BERT format. Adapted from https://github.com/ariecattan/coref
    '''
    docs_bert_tokens = []
    docs_start_end_bert = []
    docs_origin_tokens = []
    clean_start_end_dict = {}

    for doc_id in sorted(documents):
      tokens = documents[doc_id]
      bert_tokens_ids, bert_sentence_ids = [], []
      start_bert_idx, end_bert_idx = [], [] # Start and end token indices for each bert token
      original_tokens = [] 
      clean_start_end = -1 * np.ones(len(tokens), dtype=np.int)
      bert_cursor = -1
      for i, token in enumerate(tokens):
        sent_id, token_id, token_text, flag_sentence = token
        bert_token = self.tokenizer.encode(token_text, add_special_tokens=True)[1:-1]   
        if bert_token:
          bert_start_index = bert_cursor + 1
          bert_tokens_ids.extend(bert_token)
          start_bert_idx.append(bert_start_index)
          bert_cursor += len(bert_token)

          bert_end_index = bert_cursor
          end_bert_idx.append(bert_end_index)
          
          clean_start_end[i] = len(original_tokens)
          original_tokens.append([sent_id, token_id, token_text, flag_sentence])
      docs_bert_tokens.append(bert_tokens_ids)
      docs_origin_tokens.append(original_tokens)
      clean_start_end_dict[doc_id] = clean_start_end.tolist() 
      start_end = np.concatenate((np.expand_dims(start_bert_idx, 1), np.expand_dims(end_bert_idx, 1)), axis=1)
      docs_start_end_bert.append(start_end)

    return docs_origin_tokens, docs_bert_tokens, docs_start_end_bert, clean_start_end_dict
    
  def load_corpus(self, doc_path, mention_path, embed_path):
    '''
    :returns doc_ids: list of str
    :returns mentions: list of list of str
    :returns tokens_ids: [[[tokens_ids[s][i][j] for j in range(L_mention)] 
                          for i in range(N_mentions)] for s in range(N_sents)]
    :returns cluster_ids: list of list of ints
    :returns vocabs: mapping from word to int
    '''
    documents = json.load(open(doc_path))
    mention_list = json.load(open(mention_path))
    token_embeddings = np.load(embed_path)
    _, _, bert_spans, clean_span_dict = self.tokenize(documents)
    
    mention_dict = {} 
    mentions = []
    mention_heads = []
    spans = []
    postags = []
    cluster_ids = []
    embeddings = []
    doc_ids = sorted(documents)
    feat_keys = sorted(token_embeddings, key=lambda x:int(x.split('_')[-1]))
    feat_keys = [k for k in feat_keys if '_'.join(k.split('_')[:-1]) in documents]
    vocabs = {'###UNK###': 0}
    dep_rels = set()
    
    for m in mention_list:
      # Create a mapping from span to a tuple of (token, cluster_id) 
      tokens_ids = m['tokens_ids']
      span = (min(tokens_ids)-self.is_one_indexed, max(tokens_ids)-self.is_one_indexed)
      cluster_id = m['cluster_id']
      if not m['doc_id'] in mention_dict:
        mention_dict[m['doc_id']] = {}
      mention_dict[m['doc_id']][span] = cluster_id
    
    for doc_id, feat_id, bert_span in zip(doc_ids[:20], feat_keys, bert_spans): # XXX
      if not doc_id in mention_dict:
        continue
      mentions.append([])
      postags.append([])
      mention_heads.append([])
      spans.append([])
      cluster_ids.append([])
      embeddings.append([])
      
      tokens = [t[2] for t in documents[doc_id]]
      postag = [t[1] for t in nltk.pos_tag(tokens)]
      '''
      instance = self.dep_parser._dataset_reader.text_to_instance(tokens, postags)
      parse = self.dep_parser.predict_instance(instance)
      dep_head = parse['predicted_heads']
      dep_rel = parse['predicted_dependencies']
      dep_rels.update(dep_rel)
      '''
      for span in sorted(mention_dict[doc_id])[:self.max_num_mentions]:
        cluster_id = mention_dict[doc_id][span]
        mention_token = tokens[span[0]:span[1]+1][:self.max_mention_length]
        mention_tag = postag[span[0]:span[1]+1][:self.max_mention_length]

        clean_start, clean_end = clean_span_dict[doc_id][span[0]], clean_span_dict[doc_id][span[1]]
        bert_start, bert_end = bert_span[clean_start, 0], bert_span[clean_end, 1]
        mention_emb = token_embeddings[feat_id][bert_start:bert_end+1]
        
        '''
        if span[0] != span[1]:
          mention_heads[-1].append([])
          for x_idx, x in enumerate(range(span[0], span[1]+1)):
            if dep_head[x] - 1 < span[0] or dep_head[x] - 1 > span[1]:
              print('Head for mention {} at [{}, {}] is {} with dep head = {}'.format(' '.join(mention_token), span[0], span[1], tokens[x], dep_head[x] - 1))
              mention_heads[-1][-1].append(x_idx)
        else:
          mention_heads[-1].append([0])
        '''
        for t in mention_token:
          if not t in vocabs:
            vocabs[t] = len(vocabs)

        mentions[-1].append(mention_token)
        postags[-1].append(mention_tag)
        embeddings[-1].append(mention_emb)
        spans[-1].append(span)
        cluster_ids[-1].append(cluster_id)
    
    logger.info('Number of documents = {}'.format(len(mentions)))
    logger.info('Vocab size = {}'.format(len(vocabs)))
    logger.info('Dependency relations = {}'.format(dep_rels))
    return mentions, postags, embeddings, spans, cluster_ids, vocabs

  def is_compatible(self, first_tag, second_tag):
    if first_tag[:2] == second_tag[:2]:
      return True
    elif first_tag[:2] in ['NN', 'VB', 'PR'] and second_tag[0] in ['NN', 'VB', 'PR']:
      return True
    return False

  def similarity(self, first_embs, second_embs):
    sim_map = first_embs.T @ second_embs
    return sim_map.max(axis=1).mean() + sim.max(axis=0).mean()
    
  def fit(self, n_epochs=10):
    best_f1 = 0.
    for epoch in range(n_epochs):
      N_c = [np.zeros((len(sent), len(sent))) for sent in self.mentions]
      N_c_all = self.alpha0 * np.ones((self.max_num_mentions, self.max_num_mentions))
      N_mc = [np.zeros((len(sent), len(sent), self.max_mention_length, self.max_mention_length)) for sent in self.mentions]
      N_m = self.alpha0 * np.ones((len(self.vocabs), len(self.vocabs)))
      N_l = self.alpha0 * np.ones((self.max_mention_length, self.max_mention_length))

      # Compute counts
      for sent_idx, (sent, postag, embedding) in enumerate(zip(self.mentions, self.postags, self.embeddings)):
        for second_idx in range(min(len(sent), self.max_num_mentions)):
          for first_idx in range(second_idx):
            for second_token_idx, (second_token, second_tag) in enumerate(zip(sent[second_idx], postag[second_idx])):
              for first_token_idx, (first_token, first_tag) in enumerate(zip(sent[first_idx], postag[first_idx])): 
                if self.is_compatible(first_tag, second_tag):
                  first = self.vocabs[first_token]
                  second = self.vocabs[second_token]
                  N_mc[sent_idx][first_idx, second_idx, first_token_idx, second_token_idx] = self.mention_probs[first][second]
            N_c[sent_idx][first_idx, second_idx] = self.coref_probs[first_idx][second_idx] *\
                                                   self.compute_mention_pair_prob(sent[first_idx], 
                                                                                  sent[second_idx],
                                                                                  postag[first_idx],
                                                                                  postag[second_idx])
            N_c[sent_idx][first_idx, second_idx] *= np.exp(self.similarity(embedding[first_idx], embedding[second_idx]))
            N_mc[sent_idx][first_idx, second_idx] /= N_mc[sent_idx][first_idx, second_idx].sum(axis=0) + EPS
        N_c[sent_idx] /= N_c[sent_idx].sum(axis=0) + EPS
        N_c_all[:len(sent), :len(sent)] += N_c[sent_idx]
        
        for second_idx in range(min(len(sent), self.max_num_mentions)):
          for first_idx in range(second_idx):
            for first_token_idx, second_token in enumerate(sent[second_idx]):
              for second_token_idx, first_token in enumerate(sent[first_idx]):
                first = self.vocabs[first_token]
                second = self.vocabs[second_token]
                N_m[first, second] += N_c[sent_idx][first_idx, second_idx] *\
                                      N_mc[sent_idx][first_idx, second_idx, first_token_idx, second_token_idx]
            N_l[len(sent[first_idx])-1][len(sent[second_idx])-1] += N_c[sent_idx][first_idx, second_idx]
            
      # Update mention probs     
      self.mention_probs = N_m / N_m.sum(axis=-1, keepdims=True)

      # Update coref probs
      self.coref_probs = N_c_all / N_c_all.sum(axis=0)
      
      # Update length probs
      self.length_probs = N_l / N_l.sum(axis=-1, keepdims=True) 
      
      # Log stats
      logger.info('Epoch {}, log likelihood = {:.3f}'.format(epoch, self.log_likelihood()))

      # Compute pairwise coreference scores
      pairwise_f1 = self.evaluate(self.doc_path,
                                  self.mention_path, 
                                  self.out_path)
      if pairwise_f1 > best_f1:
        best_f1 = pairwise_f1
        np.save(os.path.join(self.out_path, '../translate_probs_{}.npy'.format(epoch)), self.mention_probs)
        vocabs = sorted(self.vocabs, key=lambda x:int(self.vocabs[x]))
        with open(os.path.join(self.out_path, '../translate_probs_top200.txt'), 'w') as f:
          top_idxs = np.argsort(-N_m.sum(axis=-1))[:200]
          for v_idx in top_idxs:
            v = vocabs[v_idx]
            p = self.mention_probs[v_idx] 
            top_coref_idxs = np.argsort(-p)[:100]
            for c_idx in top_coref_idxs:
              f.write('{} -> {}: {}\n'.format(v, vocabs[c_idx], p[c_idx]))
            f.write('\n')
              
  def log_likelihood(self):
    ll = -np.inf 
    N = len(self.mentions)
    for sent_idx, (mentions, postags, embeddings) in enumerate(zip(self.mentions, self.postags, self.embeddings)):
      if sent_idx == 0:
        ll = 0.
      
      for m_idx, (second, second_tag, second_emb) in enumerate(zip(mentions, postags, embeddings)):
        second_tag = postags[m_idx]
        p_sent = 0.
        for m_idx2, (first, first_tag, first_emb) in enumerate(zip(mentions[:m_idx], postags[:m_idx], embeddings[:m_idx])):
          p_sent += self.coref_probs[m_idx2, m_idx] *\
                    self.compute_mention_pair_prob(first, second, first_tag, second_tag) *\
                    np.exp(self.similarity(first_emb, second_emb))
          
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
    for m_idx, (mention, span, postag, embed) in enumerate(zip(mentions, spans, postags, embeddings)):
      if m_idx == 0:   
        cluster_labels[m_idx] = 0

      second = mention
      second_tag = postag
      second_emb = embed
      scores = np.asarray([self.compute_mention_pair_prob(first, second, first_tag, second_tag)\
                           * np.exp(self.similarity(first_emb, second_emb))
                           for first, first_tag, first_emb in zip(mentions[:m_idx+1], postags[:m_idx+1], embeddings[:m_idx+1])])
      scores = self.coref_probs[:m_idx+1, m_idx] * scores
      a_idx = np.argmax(scores)

      # If antecedent is itself, create a new cluster; otherwise
      # assign to antecedent's cluster
      if a_idx == m_idx:
        c_idx = len(clusters)
        clusters[c_idx] = [m_idx]
        cluster_labels[m_idx] = c_idx 
      else:      
        c_idx = cluster_labels[a_idx]
        cluster_labels[m_idx] = c_idx
        clusters[c_idx].append(m_idx)

    return clusters, cluster_labels

  def evaluate(self, doc_path, mention_path, embed_path, out_path):
    if not os.path.exists(out_path):
      os.makedirs(out_path)

    documents = json.load(open(doc_path, 'r')) 
    doc_ids = sorted(documents)
    mentions_all,\
    postags_all,\
    embeddings_all,\
    spans_all,\
    cluster_ids_all, _ = self.load_corpus(doc_path, mention_path, embed_path)
    
    preds = []
    golds = []
    for doc_id, mentions, postags, embeddings, spans, cluster_ids in zip(doc_ids, mentions_all, postags_all, embeddings_all, spans_all, cluster_ids_all):
      # Compute pairwise F1
      clusters, cluster_labels = self.predict(mentions, spans, postags, embeddings)
      pred = [cluster_labels[f] == cluster_labels[s] for f, s in combinations(range(len(spans)), 2)]
      pred = np.asarray(pred)

      gold = [cluster_ids[f] != 0 and cluster_ids[s] != 0 and cluster_ids[f] == cluster_ids[s] for f, s in combinations(range(len(spans)), 2)]
      gold = np.asarray(gold)
      preds.extend(pred)
      golds.extend(gold)

      # Save output files to CoNLL format
      document = {doc_id:documents[doc_id]}
      spans = np.asarray(spans)
      write_output_file(document, clusters,
                        [doc_id]*len(spans),
                        spans[:, 0].tolist(),
                        spans[:, 1].tolist(),
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

  def compute_mention_pair_prob(self, first, second, first_tags, second_tags):
    p_mention = self.length_probs[len(first)-1][len(second)-1]
    for second_token, second_tag in zip(second, second_tags):
      p_token = 0.
      for first_token, first_tag in zip(first, first_tags):
        if self.is_compatible(first_tag, second_tag):
          first_idx = self.vocabs.get(first_token, 0)
          second_idx = self.vocabs.get(second_token, 0)
          p_token += self.mention_probs[first_idx][second_idx]
      p_mention *= p_token
    return p_mention


if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--dataset', choices={'ecb', 'video_m2e2'}, default='ecb')
  parser.add_argument('--mention_type', choices={'events', 'entities', 'mixed'}, default='mixed')
  parser.add_argument('--exp_dir', default=None)
  args = parser.parse_args()
  
  dataset = args.dataset
  coref_type = args.mention_type
  doc_path_train = 'data/{}/mentions/train.json'.format(dataset)
  mention_path_train = 'data/{}/mentions/train_{}.json'.format(dataset, coref_type)
  embed_path_train = 'data/{}/mentions/train_bert_embeddings.npz'.format(dataset)
  doc_path_test = 'data/{}/mentions/test.json'.format(dataset)
  mention_path_test = 'data/{}/mentions/test_{}.json'.format(dataset, coref_type)
  embed_path_test = 'data/{}/mentions/test_bert_embeddings.npz'.format(dataset)
  
  out_path = args.exp_dir
  if not args.exp_dir:
    out_path = 'models/smt_{}/pred_{}'.format(dataset, coref_type)
    
  model = SMTCoreferencer(doc_path_train, mention_path_train, embed_path_train, out_path=out_path+'_train', config={'is_one_indexed': dataset == 'ecb'})
  model.fit(5)
  model.evaluate(doc_path_test, mention_path_test, embed_path_test, out_path=out_path+'_test')