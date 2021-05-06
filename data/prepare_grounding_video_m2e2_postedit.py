# -*- coding:utf-8 -*-
import json
import os
import pickle
import codecs
import spacy
import functools
import numpy as np

EVENTS = ['Contact.Broadcast', 'Justice.Sentence', 'Conflict.Demonstrate', 'Conflict.Attack', 'Life.Marry', 'Justice.Extradite', 'Justice.Sue', 'Transaction.TransferOwnership', 'Justice.Fine', 'Personnel.Elect', 'Contact.Contact', 'Justice.ReleaseParole', 'Personnel.EndPosition', 'Contact.Meet', 'Manufacture.Artifact', 'Life.BeBorn', 'Personnel.StartPosition', 'Transaction.TransferMoney', 'Personnel.Nominate', 'Life.Injure', 'Business.End', 'Movement.TransportArtifact', 'Justice.TrialHearing', 'Justice.ArrestJail', 'Business.DeclareBankruptcy', 'Justice.ChargeIndict', 'Movement.TransportPerson', 'Justice.Convict', 'Contact.Correspondence', 'Transaction.Transaction', 'Business.Start', 'Life.Die']
EVENTS = [e.split('.')[-1] for e in EVENTS]
PUNCT = [',', '.', '?', '\'', '\"', '!']
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))
def int_overlap(a1, b1, a2, b2):
  '''Checks whether two intervals overlap'''
  if b1 < a2 or b2 < a1:
    return False
  return True

class Token:
  def __init__(self, text, start_char, end_char):
    self.text = text
    self.start_char = start_char
    self.end_char = end_char

class Article:
  def __init__(self, rawtext, annotation):
    self.annotation = annotation
    self.tokens = []
    self.sentences = []
    self.postags = []
    self.tokenize(rawtext)
    self.ners = []
    self.ent_ids_to_ner_ids = {}
    self.get_entities()
    self.events = []
    self.event_ids_to_trigger_ids = {}
    self.get_events()
    self.entity_clusters = []
    self.event_clusters = []
    self.get_clusters()
	
  def tokenize(self, text):
    doc = nlp(text, disable=['parser', 'ner', 'textcat'])
    char_pos = 0
    for sent in doc.sents:
      self.sentences.append(Token(sent.text, char_pos, char_pos+len(sent.text)-1))
      self.tokens.append([])
      self.postags.append([])
      for token in sent:
        self.tokens[-1].append(Token(token.text, char_pos, char_pos+len(token.text)-1))
        self.postags[-1].append(token.tag_)
        char_pos += len(token.text)
        if token.whitespace_:
          char_pos += 1 

  def get_entities(self):
    '''Extract a list of mappings from mention id to entity mentions and a list of mappings from entity id to mention ids'''
    event_ids = []
    for line in self.annotation:
      parts = line.split()
      if parts[1] == 'EVENT':
        event_ids.append(parts[0])
      elif parts[0][0] == 'E':
        event_ids.append(parts[1].split(':')[-1])
 
    for sent_idx in range(len(self.sentences)):
      self.ners.append({})
      for line in self.annotation:
        parts = line.split()
        if parts[0][0] == 'T' and not parts[1] in EVENTS: # Check if the annotation is a mention
          # print('Currently reading {} {} {} {} for sent idx {}'.format(parts[0], parts[1], parts[2], parts[3], sent_idx)) # XXX
          mention_id, ner, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])-1 
          ne_start = None
          ne_end = None
          for idx, token in enumerate(self.tokens[sent_idx]):
            if token.text in PUNCT:
              continue
            if int_overlap(start, end, token.start_char, token.end_char):
              if ne_start is None:
                ne_start = idx
              ne_end = idx
          if ne_start is None: # Skip if the mention is outside current sentence
            continue 
          name = [t.text for t in self.tokens[sent_idx][ne_start:ne_end+1]]
          ne = {'start': ne_start,
                'end': ne_end,
                'entity-type': ner,
                'text': ' '.join(name)}
          self.ners[sent_idx][mention_id] = ne

  def get_events(self):
    '''Extract a list of mapping from event mention id to event mentions'''
    # Extract triggers
    triggers = []
    self.entity_triggers = dict()
    for sent_idx in range(len(self.sentences)):
      triggers.append({})
      for line in self.annotation:
        parts = line.split()
        if parts[0][0] == 'T' and parts[1] in EVENTS:
            mention_id, event_type, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])-1
            t_start = None
            t_end = None
            for idx, token in enumerate(self.tokens[sent_idx]):
              if token.text in PUNCT:
                continue
              if int_overlap(start, end, token.start_char, token.end_char):
                if t_start is None:
                  t_start = idx
                t_end = idx

            if t_start is None:
              continue
            name = [t.text for t in self.tokens[sent_idx][t_start:t_end+1]]
            triggers[sent_idx][parts[0]] = {'start': t_start, 
                                            'end': t_end,
                                            'start_char': start,
                                            'end_char': end,
                                            'event-type': event_type,
                                            'text': ' '.join(name)}
    
      for line in self.annotation:
        parts = line.split() 
        if parts[0][0] == 'T' and parts[1] == 'Event': # Deal triggers annotated as entities    
          m_id, start, end = parts[0], int(parts[2]), int(parts[3])-1 
          for e_id, e in triggers[sent_idx].items():
            if e['start_char'] == start and e['end_char'] == end: 
              self.entity_triggers[m_id] = e_id
              print(f'Trigger {e_id} and entity {m_id} are the same') # XXX
            
    for sent_idx in range(len(self.sentences)):
      self.events.append({})
      for line in self.annotation:
        parts = line.split()        
        if parts[0][0] == 'E': # Check if the annotation is an event
          event_id = parts[0]
          event_type, trigger_id = parts[1].split(':')

          if not trigger_id in triggers[sent_idx]: # Skip if the trigger is not in the current sentence
            continue
          
          for m_id, e_id in self.entity_triggers.items(): # Update entity trigger mapping to event ids
            if e_id == trigger_id:
              self.entity_triggers[m_id] = event_id

          event = {'trigger': triggers[sent_idx][trigger_id],
                   'event-type': event_type,
                   'arguments': []}
          roles = parts[2:] 
          for role in roles: 
            role_type, role_id = role.split(':') 
            if not role_id in self.ners[sent_idx]: # XXX Ignore event arguments
              continue
            event['arguments'].append({'role': role_type,
                                       'start': self.ners[sent_idx][role_id]['start'],
                                       'end': self.ners[sent_idx][role_id]['end'],
                                       'text': self.ners[sent_idx][role_id]['text']})
          self.events[sent_idx][event_id] = event
        
  def get_clusters(self):
    ''' Extract a list of mapping from entity id/event id to coreferent mentions for each sentence '''
    entity_trigger_ids = list(self.entity_triggers.keys())
    has_entity_event = False

    entity_clusters = {}
    event_clusters = {}
    for line in self.annotation:
      parts = line.split()

      if parts[1] == 'EntityCoref':
        is_event = True
        for mention_id in parts[2:]:
          if not mention_id in entity_trigger_ids:
            is_event = False
            break
        
        if is_event:
          event_clusters[self.entity_triggers[parts[2]]] = [self.entity_triggers[e_id] for e_id in parts[2:]]
        else:
          entity_clusters[parts[2]] = parts[2:]
      elif parts[1] == 'EventCoref':
        event_clusters[parts[2]] = parts[2:]
    self.event_clusters = event_clusters
    self.entity_clusters = entity_clusters


def generate_json(img_id, ann, events, entities, cluster2idx, documents):
    '''
    :param img_id: str, image id of the current example
    :param ann: str, annotation text of the current example
    :param example_list: list of processed examples of the format
                         {'image': str, image id,
                          'sentence_id': str, sentence id,
                          'sentence_start': int, start char position,
                          'sentence_end': int, end char position,
                          'sentence': str, caption text,
                          'words': list, word tokens,
                          'index': list of tuple of (start char position, end char position),
                          'pos-tags': list of str of pos tags,
                          'golden-entity-mentions': list of entity mentions of:
                             {'entity-type': str,
                              'start': int,
                              'end': int,
                              'text': str},
                          'golden-event-mentions': list of event mentions with the format:
                              {'trigger': {'start': int,
                                           'end': int,
                                           'text': str}
                               'event-type': str,
                               'arguments': list of {'role': str, 'start': int, 'end': int, 'text': str},
                          'stanford-colcc': list of str of dependency parsing results '[deprel]/dep=[word idx]/gov=[head idx]',
                          'coreference': {'entities': list of list of ints, 'events': list of list of ints},
                          'mention_ids': {'entities': list of str,
                                          'events': list of str}}
    '''
    tokens = []
    sen_start = 0
    with codecs.open(ann.replace('.ann', '.txt'), 'r', encoding='utf-8') as f:
      caption = f.read()

    with codecs.open(ann, 'r', encoding='utf-8') as f:
      annotation = f.read().strip().split('\n')

    # Tokenize and preprocess the caption
    article = Article(caption, annotation)

    # Create mappings from coreference
    mention2cluster = {}
    for cluster_id, mention_ids in article.entity_clusters.items():
      global_id = '{}_{}'.format(img_id, cluster_id)
      if not global_id in cluster2idx:
        cluster2idx[global_id] = len(cluster2idx)
      for m_id in mention_ids:
        mention2cluster[m_id] = cluster2idx[global_id]
    
    for cluster_id, mention_ids in article.event_clusters.items():
      global_id = '{}_{}'.format(img_id, cluster_id)
      if not global_id in cluster2idx:
        cluster2idx[global_id] = len(cluster2idx)
      for m_id in mention_ids:
        mention2cluster[m_id] = cluster2idx[global_id]
    
    mention_mask = np.zeros(sum(len(sent) for sent in article.tokens),)
    for sent_idx in range(len(article.sentences)):
      sent_tokens = [t.text for t in article.tokens[sent_idx]]
      # Extract entity mentions
      for mention_id in sorted(article.ners[sent_idx], key=lambda x:int(x[1:])): 
        mention = article.ners[sent_idx][mention_id]
        cluster_idx = mention2cluster.get(mention_id, 0)
        mention_mask[sen_start+mention['start']:sen_start+mention['end']+1] = 1

        entities.append({'doc_id': img_id,
                         'subtopic': '0',
                         'm_id': '{}_{}'.format(img_id, mention_id),
                         'sentence_id': sent_idx,
                         'tokens_ids': list(range(sen_start+mention['start'], sen_start+mention['end']+1)),
                         'tokens': ' '.join(sent_tokens[mention['start']:mention['end']+1]),
                         'entity_type': mention['entity-type'],
                         'tags': '',
                         'lemmas': '',
                         'cluster_id': cluster_idx,
                         'cluster_desc': '',
                         'singleton': (cluster_idx == 0)}) 

      # Extract event mentions
      for event_id in sorted(article.events[sent_idx], key=lambda x:int(x[1:])):
        event = article.events[sent_idx][event_id]['trigger']
        arguments = [{'start': sen_start+a['start'],
                      'end': sen_start+a['end'],
                      'role': a['role'],
                      'text': a['text']} for a in article.events[sent_idx][event_id]['arguments']] 

        cluster_idx = mention2cluster.get(event_id, 0)

        events.append({'doc_id': img_id,
                       'subtopic': '0',
                       'm_id': '0',
                       'sentence_id': sent_idx,
                       'tokens_ids': list(range(sen_start+event['start'], sen_start+event['end']+1)),
                       'tokens': ' '.join(sent_tokens[event['start']:event['end']+1]),
                       'arguments': arguments,
                       'event_type': event['event-type'],
                       'tags': '',
                       'lemmas': '',
                       'cluster_id': cluster_idx,
                       'cluster_desc': '',
                       'singleton': (cluster_idx == 0)})
    
      # Extract tokens
      for token_idx, token in enumerate(article.tokens[sent_idx]):
        tokens.append([sent_idx, sen_start+token_idx, token.text, (int(mention_mask[token_idx]) > 0)])
      sen_start = len(tokens)
    documents[img_id] = tokens

    return events, entities, cluster2idx, documents

def generate_json_all(pair_list, out_prefix):
    events = []
    entities = []
    cluster2idx = {'###SINGLETON###': 0}
    documents = {}

    for image_id, ann_file in pair_list:
      print(image_id, ann_file)
      events, entities, cluster2idx, documents = generate_json(image_id, ann_file, events, entities, cluster2idx, documents)
    
    n_event_clusters = len(set([e['cluster_id'] for e in events if e['cluster_id']]))
    n_entity_clusters = len(set([e['cluster_id'] for e in entities if e['cluster_id']]))
    n_clusters = len(cluster2idx)
    print(f'Number of event mentions: {len(events)}, number of entity mentions: {len(entities)}')
    print(f'Number of event clusters: {n_event_clusters}, number of entity clusters: {n_entity_clusters}, total number of clusters: {n_clusters}')
    json.dump(documents, codecs.open(out_prefix+'.json', 'w', 'utf-8'), indent=2)
    json.dump(entities, codecs.open(out_prefix+'_entities.json', 'w', 'utf-8'), indent=2)
    json.dump(events, codecs.open(out_prefix+'_events.json', 'w', 'utf-8'), indent=2)
    json.dump(entities+events, codecs.open(out_prefix+'_mixed.json', 'w', 'utf-8'), indent=4, sort_keys=True)
      

def main(grounding_dir, img_dir, m2e2_caption, m2e2_annotation_dir, out_prefix='video_m2e2'):
    '''
    :param grounding_dir: str, directory name for the grounding meta info files
    :param img_dir: str, directory of the images/videos
    :param m2e2_caption: str, json file name storing the dictionary with the format:
                         [short desc]: {'id': [caption text], 'long_desc': [url to the image]}
    :param m2e2_annotation_dir: str, directory name of the annotations
    '''
    m2e2_image_caption = json.load(open(m2e2_caption))
    pairs = list()
    count = 0
     
		# Filter out unannotated examples
    ann_files = []
    for ann_file in os.listdir(os.path.join(m2e2_annotation_dir)):
      if ann_file.split('.')[-1] != 'ann':
        continue
      with open(os.path.join(m2e2_annotation_dir, ann_file), 'r') as f:
        if len(f.readlines()) > 0:
          ann_files.append(ann_file)
    
    keys = sorted(m2e2_image_caption) 
    for ann_file in ann_files: # XXX
      caption_dict = m2e2_image_caption[keys[int(ann_file.split('.')[0])]] 
      youtube_id = caption_dict['id'].split('v=')[-1]

      # if not os.path.isfile(os.path.join(img_dir, youtube_id+'.mp4')):
      #   os.system('youtube-dl -f mp4 -i {}'.format(caption_dict['id']))
      pairs.append((youtube_id, os.path.join(m2e2_annotation_dir, ann_file)))
    generate_json_all(pairs, out_prefix)
    

if __name__ == '__main__':
  grounding_dir = ''
  img_dir = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/MultimediaEventCoreference/m2e2/data/video_m2e2/videos/'
  m2e2_caption = 'video_m2e2/video_m2e2.json' # '/ws/ifp-53_2/hasegawa/lwang114/fall2020/MultimediaEventCoreference/m2e2/data/video_m2e2/video_m2e2.json'
  m2e2_annotation_dir = '../brat/brat-v1.3_Crunchy_Frog/data/video_m2e2_oneie/train/'
  main(grounding_dir, img_dir, m2e2_caption, m2e2_annotation_dir, out_prefix='traintest')
