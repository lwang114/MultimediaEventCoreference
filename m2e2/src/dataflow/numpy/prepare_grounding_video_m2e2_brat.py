# -*- coding:utf-8 -*-
import json
import os
import pickle
import codecs
import spacy
import functools
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

NE = ['ABS', 'AML', 'BAL', 'BOD', 
      'COM', 'FAC', 'GPE', 'INF',
      'LAW', 'LOC', 'MHI', 'MON',
      'NAT', 'ORG', 'PER', 'PLA',
      'PTH', 'RES', 'SEN', 'SID',
      'TTL', 'VAL', 'VEH', 'WEA', 'UNK']
UNK_EVENT = 'EVENT'
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))
dep_parser = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz')
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
	
  def tokenize(self, text): # TODO Deal with ''read more'' and non-ascii symbols
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
        if parts[0][0] == 'T' and parts[1] in NE: # Check if the annotation is a mention
          # print('Currently reading {} {} {} {} for sent idx {}'.format(parts[0], parts[1], parts[2], parts[3], sent_idx)) # XXX
          mention_id, ner, start, end = parts[0], parts[1], int(parts[2]), int(parts[3]) 
          ne_start = None
          ne_end = None
          for idx, token in enumerate(self.tokens[sent_idx]):
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
    for sent_idx in range(len(self.sentences)):
      triggers.append({})
      for line in self.annotation:
        parts = line.split()
        if parts[0][0] == 'T' and not parts[1] in NE:
            mention_id, event_type, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])
            t_start = None
            t_end = None
            for idx, token in enumerate(self.tokens[sent_idx]):
              if int_overlap(start, end, token.start_char, token.end_char):
                if t_start is None:
                  t_start = idx
                t_end = idx

            if t_start is None:
              continue
            name = [t.text for t in self.tokens[sent_idx][t_start:t_end+1]]
            triggers[sent_idx][parts[0]] = {'start': t_start, 
                                            'end': t_end,
                                            'text': ' '.join(name)} 
	  # Extract event types and arguments
    for sent_idx in range(len(self.sentences)):
      self.events.append({})
      for line in self.annotation:
        parts = line.split()        
        if parts[0][0] == 'E': # Check if the annotation is an event
          event_id = parts[0]
          event_type, trigger_id = parts[1].split(':')
          if not trigger_id in triggers[sent_idx]: # Skip if the trigger is not in the current sentence
            continue

          event = {'trigger': triggers[sent_idx][trigger_id],
                   'trigger_id': trigger_id,
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
        
    # Include events with no arguments (isolated event)
    for sent_idx in range(len(self.sentences)):
      for trigger_id, trigger_info in triggers[sent_idx].items():
        if len(self.events[sent_idx]) == 0 or\
          functools.reduce(lambda x, y: x and y,\
          map(lambda x: x[1]['trigger']['start'] != trigger_info['start'] and 
                        x[1]['trigger']['end'] != trigger_info['end'], 
            self.events[sent_idx].items())):
          print('find an isolated event {}'.format(trigger_id))
          self.events[sent_idx][trigger_id] = {'trigger': triggers[sent_idx][trigger_id],
                                               'trigger_id': trigger_id,
                                               'event-type': UNK_EVENT,
                                               'arguments': []}


  def get_clusters(self):
    ''' Extract a list of mapping from entity id/event id to coreferent mentions for each sentence '''
    trigger2events = []
    for sent_idx in range(len(self.sentences)):
      trigger2events.append({})
      # Create a mapping from EVENT entity id to event trigger id
      for line in self.annotation:
        parts = line.split()
        if parts[1] == 'EVENT':
          trigger_id, start, end = parts[0], int(parts[2]), int(parts[3])
          t_start = None
          t_end = None 
          for idx, token in enumerate(self.tokens[sent_idx]):
            if int_overlap(start, end, token.start_char, token.end_char):
              if t_start is None:
                t_start = idx
              t_end = idx
          if t_start is None:
            continue
          
          for event_id, event in self.events[sent_idx].items():
            if t_start == event['trigger']['start'] and t_end == event['trigger']['end']:
              trigger2events[sent_idx][trigger_id] = event_id

      entity_cluster = {}
      event_cluster = {}
      for line in self.annotation:
        parts = line.split()
        
        if parts[1] == 'Alias': # Check if the annotation is a coreference
          for mention_id in parts[2:]:
            if mention_id in trigger2events[sent_idx]:
              if not parts[2] in event_cluster:
                event_cluster[parts[2]] = [trigger2events[sent_idx][mention_id]]
              else:
                event_cluster[parts[2]].append(trigger2events[sent_idx][mention_id])
            elif mention_id in self.ners[sent_idx]:
              if not parts[2] in entity_cluster:
                entity_cluster[parts[2]] = [mention_id]
              else:
                entity_cluster[parts[2]].append(mention_id)
      
      self.event_clusters.append(event_cluster)
      self.entity_clusters.append(entity_cluster)

def generate_json(img_id, ann, example_list):
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
    with codecs.open(ann.replace('.ann', '.txt'), 'r', encoding='utf-8') as f:
      caption = f.read()

    with codecs.open(ann, 'r', encoding='utf-8') as f:
      annotation = f.read().strip().split('\n')

    # Tokenize and preprocess the caption
    article = Article(caption, annotation)
    for idx in range(len(article.sentences)):
      sent_id = '{}_{}'.format(img_id, idx)
      sen_obj = dict()
      sen_obj['image'] = img_id
      sen_obj['sentence_id'] = sent_id
      sen_obj['sentence_start'] = article.sentences[idx].start_char
      sen_obj['sentence_end'] = article.sentences[idx].end_char
      sen_obj['sentence'] = article.sentences[idx].text
      sen_obj['words'] = list()
      sen_obj['index'] = list()
      sen_obj['pos-tags'] = list()
      sen_obj['stanford-colcc'] = list()
      sen_obj['golden-entity-mentions'] = list()
      sen_obj['golden-event-mentions'] = list()
      sen_obj['mention_ids'] = {'entities': {}, 'events': {}}
      sen_obj['coreference'] = {'entities': {}, 'events': {}}

      # Save tokenized sentence
      for token in article.tokens[idx]:
        sen_obj['words'].append(token.text)
        sen_obj['index'].append([token.start_char, token.end_char])

      # Save pos-tags
      sen_obj['pos-tags'].extend(article.postags[idx])

      # Save entity labels
      sen_obj['mention_ids']['entities'] = sorted(article.ners[idx], key=lambda x:int(x[1:])) 
      entities = [article.ners[idx][k] for k in sorted(article.ners[idx], key=lambda x:int(x[1:]))]
      sen_obj['golden-entity-mentions'].extend(entities)
      entid2idx = {ent_id:i for i, ent_id in enumerate(sen_obj['mention_ids']['entities'])}

      # Save event labels
      sen_obj['mention_ids']['events'] = sorted(article.events[idx], key=lambda x:int(x[1:]))
      events = [article.events[idx][k] for k in sen_obj['mention_ids']['events']]
      eventid2idx = {event_id:i for i, event_id in enumerate(sen_obj['mention_ids']['events'])} 
      sen_obj['golden-event-mentions'].extend(events)

      # Save coreference labels
      ## Entity clusters
      for cluster_id in sorted(article.entity_clusters[idx]):
        mention_ids = article.entity_clusters[idx][cluster_id] 
        mention_idxs = [(entid2idx[m_id], m_id) for m_id in mention_ids]
        sen_obj['coreference']['entities'][cluster_id] = mention_idxs       
      ## Event clusters
      for cluster_id in article.event_clusters[idx]:
        mention_ids = article.event_clusters[idx][cluster_id]
        mention_idxs = [(eventid2idx[m_id], m_id) for m_id in mention_ids]
        sen_obj['coreference']['events'][cluster_id] = mention_idxs

      # Dependency parsing 
      instance = dep_parser._dataset_reader.text_to_instance(sen_obj['words'], sen_obj['pos-tags'])
      parse = dep_parser.predict_instance(instance)
      for word_idx, (head, deprel) in enumerate(zip(parse['predicted_heads'], parse['predicted_dependencies'])):
        sen_obj['stanford-colcc'].append('{}/dep={}/gov={}'.format(deprel, word_idx, head-1))
      example_list.append(sen_obj)

    return example_list 

def generate_json_all(pair_list):
    example_list = []
    for image_id, ann_file in pair_list:
      print(image_id, ann_file)
      example_list = generate_json(image_id, ann_file, example_list)
    return example_list 

def download_video(m2e2_caption):
    m2e2_image_caption = json.load(open(m2e2_caption))
    for _, caption_dict in m2e2_image_caption.items():
      youtube_id = caption_dict['id'].split('v=')[-1]
      video_name = os.path.join(img_dir, youtube_id+'.mp4')
      if not os.path.isfile(video_name):
        print(caption_dict['id'], youtube_id, youtube_id == '7WMebV5qt3s')
        os.system('youtube-dl -f mp4 -R 1 -o {} {}'.format(video_name, caption_dict['id']))
      
      if not os.path.isfile(video_name):
        print(caption_dict['id'], youtube_id, youtube_id == '7WMebV5qt3s')
        os.system('youtube-dl -R 1 -o {} {}'.format(video_name, caption_dict['id']))

      

def main(grounding_dir, img_dir, m2e2_caption, m2e2_annotation_dir, out_prefix='m2e2'):
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
    
		# Create a list of (youtube_id, ann_file); download images/videos to img_dir
    # download_video(m2e2_caption) 
 
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

    result = generate_json_all(pairs)
    _file = codecs.open(os.path.join(grounding_dir, out_prefix + '.json'), 'w', 'utf-8')  
    json.dump(result, _file, indent=2)
    

if __name__ == '__main__':
  grounding_dir = ''
  img_dir = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/MultimediaEventCoreference/m2e2/data/video_m2e2/videos/'
  m2e2_caption = '../../../data/video_m2e2/video_m2e2.json' # '/ws/ifp-53_2/hasegawa/lwang114/fall2020/MultimediaEventCoreference/m2e2/data/video_m2e2/video_m2e2.json'
  m2e2_annotation_dir = '../../../../brat/brat-v1.3_Crunchy_Frog/data/video_m2e2/unannotated/'
  main(grounding_dir, img_dir, m2e2_caption, m2e2_annotation_dir)
