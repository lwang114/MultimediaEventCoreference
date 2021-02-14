NEW = '###NEW###'
class SMTCoreferencer:
  def __init__(self, doc_path, mention_path, out_path, config):
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
    self.out_path = out_path
    self.max_mention_length = config.get('max_mention_length', 20)
    self.max_num_mentions = config.get('max_num_mentions', 200)
    self.is_one_indexed = config.get('is_one_indexed', False)
    if not os.path.exists(self.out_path):
      os.makedirs(self.out_path)
    
    self.mentions,\
    self.postags,\
    self.tokens_ids,\
    self.cluster_ids,\
    self.vocabs = self.load_corpus(doc_path, mention_path)
    logger.info('Number of documents = {}'.format(len(self.mentions)))
    logger.info('Vocab size = {}'.format(len(self.vocabs)))

    self.vocabs[NEW] = len(self.vocabs)
    self.mention_probs = 1. / len(self.vocabs) * np.ones((2, len(self.vocabs), len(self.vocabs)))
    self.alpha0 = EPS

  def load_corpus(self.doc_path, mention_path):
    '''
    :returns doc_path: str,
    :returns mention_path: str    
    '''
    documents = json.load(open(doc_path))
    mention_list = json.load(open(mention_path))
    mention_dict = {}
    mentions = []
    spans = []
    postags = []
    cluster_ids = []
    vocabs = {'###UNK###': 0}
    
    for m in mention_list:
      # Create a mapping from span to cluster id
      tokens_ids = m['tokens_ids']
      span = (min(tokens_ids)-self.is_one_indexed, max(tokens_ids)-self.is_one_indexed)
      cluster_id = m['cluster_id']
      if not m['doc_id'] in mention_dict:
        mention_dict[m['doc_id']] = {}
      mention_dict[m['doc_id']][span] = cluster_id

    for doc_id in sorted(documents): # XXX
      if not doc_id in mention_dict:
        continue
      mentions.append([NEW])
      postags.append([NEW])
      spans.append([-1, -1])
      cluster_ids.append([-1])

      tokens = [t[2] for t in documents[doc_id]]
      postag = [t[1] for t in nltk.pos_tag(tokens)]

      for span in sorted(mention_dict[doc_id])[:self.max_num_mentions]:
        cluster_id = mention_dict[doc_id][span]
        mention_token = tokens[span[0]:span[1]+1][:self.max_mention_length]
        mention_tag = postag[span[0]:span[1]+1][:self.max_mention_length]
        if not ' '.join(mention_token) in vocabs:
          vocabs[' '.join(mention_token)] = len(vocabs)

        mentions[-1].append(mention_token)
        postags[-1].append(mention_tag)
        spans[-1].append(span)
        cluster_ids[-1].append(cluster_id)
 
    return mentions, postags, spans, cluster_ids, vocabs

  def is_match(self, first, second):
    return ' '.join(first) == ' '.join(second)

  def is_compatible(self, first_tags, second_tags):
    for first_tag in first_tags:
      for second_tag in second_tags:
        if first_tag[:2] == second_tag[:2]:
          return True
        elif first_tag[:2] in [NEW, 'NN', 'VB', 'PR'] and second_tag[0] in ['NN', 'VB', 'PR']:
          return True
    return False
  
  def set_mode(self, mention, antecedents):
    for ant in antecedents:
      if is_match(ant, mention):
        return 0
    return 1

  def fit(self, n_epochs=10):
    best_f1 = 0.
    for epoch in range(n_epochs):
      N_c = [np.zeros((2, len(sent), len(sent))) for sent in self.mentions]
      N_m = [np.zeros((self.num_modes, len(sent), len(sent), self.max_mention_length, self.max_mention_length)) for sent in self.mentions]

      for sent_idx, (sent, postag) in enumerate(zip(self.mentions, self.postags)):
        for second_idx in range(min(len(sent), self.max_num_mentions)):
          mode = self.set_mode(sent[second_idx], sent[:second_idx])
          for first_idx in range(second_idx):
            if mode == 0 and self.is_match(sent[first_idx], sent[second_idx]):
              N_c[sent_idx][0, first_idx, second_idx] += 1
            elif mode == 1 and self.is_compatible(postag[first_idx], postag[second_idx]):
              first_token_idx = self.vocabs[' '.join(sent[first_idx])]
              second_token_idx = self.vocabs[' '.join(sent[second_idx])] 
              N_c[sent_idx][1, first_idx, second_idx] += self.mention_probs[1, first_token_idx, second_token_idx]
        N_c[sent_idx] /= N_c[sent_idx].sum(axis=1, keepdims=True) + EPS
         
        for second_idx in range(min(len(sent), self.max_num_mentions)):
          for first_idx in range(second_idx):
            for mode in range(2):
              first_token_idx = self.vocabs[' '.join(sent[first_idx])]
              second_token_idx = self.vocabs[' '.join(sent[second_idx])] 
              N_m[mode, first_token_idx, second_token_idx] += N_c[sent_idx][mode, first_idx, second_idx]
      # Update mention probs
      self.mention_probs = N_m / N_m.sum(axis=-1, keepdims=True)

      # Compute pairwise coreference scores
      pairwise_f1 = self.evaluate(self.doc_path,
                                  self.mention_path,
                                  self.out_path)
      
      np.save(os.path.join(self.out_path, '../translate_probs.npy'), self.mention_probs)
      if pairwise_f1 > best_f1:
        best_f1 = pairwise_f1
        np.save(os.path.join(self.out_path, '../translate_probs_best.npy'), self.mention_probs)
        with open(os.path.join(self.out_path, '../translate_probs_top200.txt'), 'w') as f:
          top_idxs = np.argsort(-N_m.sum(axis=(0, -1)))[:200]
          for v_idx in top_idxs:
            v = vocabs[v_idx]
            for mode in range(2):
              p = self.mention_probs[mode, v_idx] 
              top_coref_idxs = np.argsort(-p)[:100]
              for c_idx in top_coref_idxs:
                f.write('Mode {}: {} -> {}: {}\n'.format(mode, v, vocabs[c_idx], p[c_idx]))
              f.write('\n')
 
  def log_likelihood(self):
    ll = -np.inf
    N = len(self.mentions)
    for sent_idx, (mentions, postags) in enumerate(zip(self.mentions, self.postags)):
      if sent_idx == 0:
        ll = 0
      
      for m_idx, (second, second_tag) in enumerate(zip(mentions, postags)):
        if m_idx == 0:
          continue
        mode = self.set_mode(second, mentions[:m_idx])
        second_tag = postags[m_idx]
        
        p_sent = 0.
        for m_idx2, (first, first_tag) in enumerate(zip(mentions[:m_idx], postags[:m_idx])):
          p_sent += 1. / m_idx * self.mention_probs[mode, first, second]
        ll += 1. / N * np.log(p_sent + EPS)
    return ll

  def predict(self, mentions, spans, postags): # TODO
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
      for first_idx in range(second_idx):
        if mode == 0 and self.is_matched(mentions[first_idx], mentions[second_idx]):
          scores.append(1.)
        elif mode == 1 and self.is_compatible(postags[first_idx], postags[second_idx]):
          first_token_idx = self.vocabs.get(' '.join(mentions[first_idx]), 0)
          second_token_idx = self.vocabs.get(' '.join(mentions[second_idx]), 0)
          scores.append(self.mention_probs[first_token_idx, second_token_idx])
         
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

  def evaluate(self, doc_path, mention_path, out_path):
    if not os.path.exists(out_path):
      os.makedirs(out_path)

    documents = json.load(open(doc_path, 'r'))
    doc_ids = sorted(documents)
    mentions_all, postags_all, spans_all, cluster_ids_all, _ = self.load_corpus(doc_path, mention_path)
    preds = []
    golds = []
    for doc_id, postags, mentions, spans, cluster_ids in zip(doc_ids, postags_all, mentions_all, spans_all, cluster_ids_all):
      clusters, cluster_labels = self.predict(mentions, spans, postags)
      pred = [cluster_labels[f] == cluster_labels[s] for f, s in combinations(range(len(spans)-1), 2)]
      pred = np.asarray(pred)

      gold = [cluster_ids[f] != 0 and cluster_ids[s] != 0 and cluster_ids[f] == cluster_ids[s] for f, s in combinations(range(len(spans)-1, 2))]
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
  parser.add_argument('--dataset', choices={'ecb', 'video_m2e2'}, default='ecb')
  parser.add_argument('--mention_type', choices={'events', 'entities', 'mixed'}, default='mixed')
  parser.add_argument('--exp_dir', default=None)
  args = parser.parse_args()
  
  dataset = args.dataset
  coref_type = args.mention_type
  doc_path_train = 'data/{}/mentions/train.json'.format(dataset)
  mention_path_train = 'data/{}/mentions/train_{}.json'.format(dataset, coref_type)
  doc_path_test = 'data/{}/mentions/test.json'.format(dataset)
  mention_path_test = 'data/{}/mentions/test_{}.json'.format(dataset, coref_type)
  out_path = args.exp_dir
  if not args.exp_dir:
    out_path = 'models/smt_{}/pred_{}'.format(dataset, coref_type)
    
  model = SMTCoreferencer(doc_path_train, mention_path_train, out_path=out_path+'_train', config={'is_one_indexed': dataset == 'ecb'})
  model.fit(5)
  model.evaluate(doc_path_test, mention_path_test, out_path=out_path+'_test')
