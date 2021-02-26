class ACEDataset(Dataset):
  def __init__(self, doc_json, mention_json, preprocessor, config, split='train'):
    '''
    :param doc_json: str of filename of the documents
    :param mention_json: str of filename of the mentions
    :param label_json: str of filename of files storing the event and argument labels in a json list of dicts with keys:
        doc_id: str,
        sent_id: str,
        tokens: list of str,
        entity_mentions: list of dicts of 
        {
          id: str,
          text: str,
          start: int, start token idx
          end: int, end token idx
          entity_type: str,
        }
        event_mentions: list of dicts of
        {  
          id: str,
          text: str,
          start: int, start token idx
          end: int, end token idx
          event_type: str,
          trigger: a dict of
            {
              text: str,
              start: str,
              end: str
          arguments: list of dicts of
            {
              entity_id: str,
              text: str,
              role: str
             }
        }
    ''' 
    super(ACEDataset, self).__init__()
    self.preprocessor = preprocessor
    self.max_token_num = config.get('max_token_num', 512)
    self.max_span_num = config.get('max_span_num', 80)
    self.max_role_num = config.get('max_role_num', 10)
    self.max_frame_num = config.get('max_mention_span', 500)
    self.split = split
    self.config = config

    mentions = json.load(open(mention_json))
    self.label_dict = self.create_dict_labels(mentions)
    self.event_ids = sorted([m['m_id'] for m in mentions])
    self.doc_ids = sorted([m['doc_id'] for m in mentions])

    embed_file = '{}_glove_embeddings.npz'.format(doc_json.split('.')[0])
    self.docs_embeddings = np.load(embed_file)
    doc_to_feat = {'_'.join(k.split('_')[:-1]):k for k in sorted(self.docs_embeddings, key=lambda x:int(x.split('_')[-1]))}
    self.feat_keys = [doc_to_feat[doc_id] for doc_id in self.doc_ids]
    print('Number of event instances: {}, {}'.format(len(self.doc_ids), len(event_ids)))

  def create_dict_labels(self, mentions):
    '''
    :param label_json: filename for a list of dicts
    :return label_dict: a mapping from event id to a dict of 
    {
      'event': a mapping from (start token, end token) -> event integer label 
      'roles': a mapping from (start token, end token) -> role integer label
      'entities': a mapping from (start token, end token) -> entity integer label 
    }
    '''
    label_dict = collections.defaultdict(dict)
    for m in mentions:
      event_id = m['m_id']
      start = min(m['tokens_ids'])
      end = max(m['tokens_ids'])
      event_label = m['event_type']
      label_dict[event_id] = {
          'event': {(start, end): self.preprocessor.event_stoi.get(event_label, 0)},
          'roles': {},
          'entities': {}
        }

      for a in m['arguments']:
        start = min(a['tokens_ids'])
        end = max(a['tokens_ids'])
        role_label = a['role']
        entity_label = a['entity_type']
        label_dict[event_id]['roles'][(start, end)] = self.preprocessor.role_stoi.get(role_label, 0)
        label_dict[event_id]['entities'][(start, end)] = self.preprocessor.entity_stoi.get(entity_label, 0)
    return label_dict

  def __getitem__(self, idx):
    '''Load doc embeddings, mention mappings and labels for the document 
    :param idx: int, doc index
    :return start_mappings: FloatTensor of size (max num. spans, max num. tokens)
    :return end_mappings: FloatTensor of size (max num. spans, max num. tokens)
    :return continuous_mappings: FloatTensor of size (max num. spans, max mention span, max num. tokens)
    :return event_mapping: LongTensor of size (max num. spans,)
    :return role_mappings: LongTensor of size (max num. roles, max num. spans)
    :return entity_labels: LongTensor of size (max num. roles,) 
    :return event_labels: Scalar LongTensor
    :return role_labels: LongTensor of size (max num. roles,) 
    '''
    event_id = self.event_ids[idx]
    event_label_dict = self.label_dict[event_id]['event']
    entity_label_dict = self.label_dict[event_id]['entities']
    role_label_dict = self.label_dict[event_id]['roles']

    # Extract the original spans of the current doc
    event_spans = sorted(self.label_dict[event_id]['event'])
    role_spans = sorted(self.label_dict[event_id]['roles'])
    entity_spans = sorted(self.label_dict[entity_id]['entities'])
    event_spans = np.asarray(event_spans)
    role_spans = np.asarray(role_spans)
    entity_spans = np.asarray(entity_spans)
    candidate_start_ends = np.concatenate([event_spans, role_spans], axis=0)

    # Extract doc embedding
    feat_id = self.feat_keys[idx]
    doc_embeddings = self.docs_embeddings[feat_id]
    doc_embeddings = fix_embedding_length(torch.FloatTensor(doc_embeddings), self.max_token_num) 

    # Pad/truncate the outputs to max num. of spans
    candidate_starts = candidate_start_ends[:, 0]
    candidate_ends = candidate_start_ends[:, 1] 
    start_mappings, end_mappings, continuous_mappings, width =\
      get_all_token_mapping(candidate_starts,
                            candidate_ends,
                            self.max_token_num,
                            self.max_mention_span)
    start_mappings = fix_embedding_length(start_mappings, self.max_span_num)
    end_mappings = fix_embedding_length(end_mappings, self.max_span_num)
    continuous_mappings = fix_embedding_length(continuous_mappings, self.max_span_num)
    width = fix_embedding_length(width.unsqueeze(1), self.max_span_num).squeeze(1)

    # Extract entity, event and role labels
    event_mappings = torch.zeros((1, self.max_span_num))
    role_mappings = torch.zeros((self.max_role_num, self.max_span_num)) 
    entity_mappings = torch.zeros((self.max_role_num, self.max_span_num))
    event_label = torch.zeros((1,))
    role_labels = torch.zeros((self.max_role_num,))
    entity_labels = torch.zeros((self.max_role_num,))

    n_events = 1
    for m_idx, span in enumerate(event_spans):
      event_mappings[m_idx, m_idx] = 1.
      event_label[m_idx] = event_label_dict[span]
    
    for m_idx, span in enumerate(role_spans):
      if m_idx >= self.max_role_num:
        continue
      role_mappings[m_idx, n_events+m_idx] = 1.
      role_labels[m_idx] = role_label_dict[span]

    for m_idx, span in enumerate(entity_spans):
      if m_idx >= self.max_role_num:
        continue
      entity_mappings[m_idx, n_events+m_idx] = 1.
      entity_labels[m_idx] = entity_label_dict[span]

    doc_len = doc_embeddings.shape[0]
    span_num = len(candidate_start_ends)
    text_mask = torch.FloatTensor([1. if j < doc_len else 0 for j in range(self.max_token_num)])
    span_mask = torch.FloatTensor([1. if i < span_num else 0 for i in range(self.max_span_num)])

    return doc_embeddings,\
           start_mappings,\
           end_mappings,\
           continuous_mappings,\
           width,\
           event_labels,\
           role_labels,\
           entity_labels,\
           event_mappings,\
           role_mappings,\
           entity_mappings,\
           text_mask, span_mask

  def __len__(self):
    return len(self.event_ids)

def Preprocessor:
  def __init__(self, mention_json):
    '''
    :param data_json: filename of the meta info file
    '''
    self.event_stoi = {'###NULL###': 0}
    self.entity_stoi = {'###NULL###': 0}
    self.role_stoi = {'###NULL###': 0}

    mentions = json.load(open(mention_json))
    for m in mentions:
        event_label = m['event_type']
        if not event_label in event_stoi:
          self.event_stoi[event_label] = len(self.event_stoi)

        for a in m['arguments']:
          role = a['role']
          entity_label = a['entity_type']
          if not role in self.role_stoi:
            self.role_stoi[role] = len(self.role_stoi)

          if not entity_label in self.entity_stoi:
            self.entity_stoi[entity_label] = len(self.entity_stoi)
            
    self.n_event_types = len(self.event_stoi)
    self.n_entity_types = len(self.entity_stoi)
    self.n_role_types = len(self.role_stoi)
    print('Number of event types {}, number of role types {}, number of entity types {}'.format(self.n_event_types, self.n_role_types, self.n_entity_types))
