# -*- coding:utf-8 -*-
import json
import os
import pickle
import codecs
import spacy
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

nlp = spacy.load('en_core_web_sm')
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
    doc = nlp(text, disable=['tagger', 'parser', 'ner', 'textcat'])
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
        if parts[0][0] == 'T' and not parts[1] in event_ids: # Check if the annotation is a mention
          mention_id, ner, start, end = parts[0], parts[1], parts[2], parts[3] 
          ne_start = None
          ne_end = None
          for idx, token in enumerate(self.tokens[sent_idx]):
            if int_overlap(start, end, token.start_char, token.end_char):
              if ne_start is None:
                ne_start = idx
              ne_end = idx

          if not ne_start: # Skip if the mention is outside current sentence
            continue

          name = [t.text for t in self.tokens[ne_start:ne_end+1]]
          ne = {'start': ne_start,
                'end': ne_end,
                'entity-type': ner,
                'text': ' '.join(name)}
          self.ners[sent_idx][mention_id] = ne

  def get_events(self):
    '''Extract a list of mapping from event mention id to event mentions'''
    triggers = []
    for sent_idx in range(len(self.sentences)):
      triggers.append({})
      for line in self.annotation:
        parts = line.split()
        if parts[0][0] == 'T':
          is_entity = False
        
          if parts[0] in self.ners[sent_idx]:
            is_entity = True
            break

        if not is_entity:
          mention_id, event_type, start, end = parts[0], parts[1], parts[2], parts[3]
          t_start = None
          t_end = None
          for idx, token in enumerate(self.tokens[sent_idx]):
            if int_overlap(start, end, token.start_char, token.end_char):
              if t_start is None:
                t_start = idx
              t_end = idx

          if not t_start:
            continue
          name = [t.text for t in self.tokens[t_start:t_end+1]]
          triggers[sent_idx][parts[0]] = {'start': t_start, 
                                          'end': t_end,
                                          'text': ' '.join(name)} 
	
    for sent_idx in range(len(self.sentences)):
      self.events.append({})
      for line in self.annotation:
        parts = line.split()        
        if parts[0][0] == 'E': # Check if the annotation is an event
          event_id = parts[0]
          event_type, trigger_id = parts[1].split(':')

          event = {'trigger': triggers[trigger_id],
                   'event-type': event_type,
                   'arguments': []}
          roles = parts[2:] 
          for role in roles: 
            role_type, role_id = role.split(':') 
            if not role_id in self.entities[sent_idx]: # XXX Ignore the event arguments
              continue
            event['arguments'].append({'role': role_type,
                                       'start': entities[sent_idx][role_id]['start'],
                                       'end': entities[sent_idx][role_id]['end'],
                                       'text': entities[sent_idx][role_id]['text']})
          self.events[sent_idx][event_id] = event 

  def get_clusters(self):
    ''' Extract a list of mapping from entity id/event id to coreferent mentions for each sentence '''
    trigger2events = []
    for sent_idx in range(len(self.sentences)):
      trigger2events.append({})
      for line in self.annotation:
        parts = line.split()
        if parts[1] == 'EVENT':
          trigger_id, start, end = parts[0], parts[2], parts[3]
          t_start = None
          t_end = None 
          for idx, token in enumerate(self.tokens[sent_idx]):
            if int_overlap(start, end, token.start_char, token_char):
              if t_start is None:
                t_start = idx
              t_end = idx
          if not t_start:
            continue
             
          for event_id, event in self.events[sent_idx].items():
            if t_start == event['start'] and t_end == event['end']: # TODO Check this 
              trigger2events[sent_idx][trigger_id] = event_id
              break

      entity_cluster = {}
      event_cluster = {}
      for line in self.annotation:
        parts = line.split()
        
        if parts[1] == 'Alias': # Check if the annotation is a coreference	
          cluster[parts[0]] = []
          for mention_id in parts[2:]:
            if mention_id in trigger2events[sent_idx]: 
              event_cluster[parts[0]].append(trigger2events[sent_idx][mention_id])
            elif mention_id in self.entities[sent_idx]:
              entity_cluster[parts[0]].append(mention_id)

      
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
    :return example_list: updated list of processed examples
    '''
    with codecs.open(ann_file.replace('ann', 'txt'), 'r', encoding='utf-8') as f:
      caption = f.read()

    with codecs.open(ann_file, 'r', encoding='utf-8') as f:
      annotation = f.read().strip().split('\n')

    # Tokenize and preprocess the caption
    article = Article(caption, annotation)
    for idx in range(len(article.sentences)):
      sent_id = '{}__{}'.format(sent_id, idx)
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
      sen_obj['mention_ids']['entities'] = sorted(article.entities[idx], lambda x:int(x[1:])) 
      entities = [article.ners[idx][k] for k in sorted(article.ners[idx], lambda x:int(x[1:]))]
      sen_obj['golden-entity-mentions'].extend(entities)
      entid2idx = {ent_id:i for i, ent_id in enumerate(sen_obj['mention_ids']['entities'])}

      # Save event labels
      sen_obj['mention_ids']['events'] = sorted(article.events[idx], lambda x:int(x[1:]))
      events = [article.events[idx][k] for k in sen_obj['mention_ids']['events']]
      eventid2idx = {event_id:i for i, event_id in enumerate(sen_obj['mention_ids']['events'])} 
      sen_obj['golden-event-mentions'].extend(events)

      # Save coreference labels
      for cluster_idx, mention_ids in enumerate(self.entity_clusters[idx]):   
        mention_idxs = [entid2idx[m_id] for m_id in mention_ids]
        sen_obj['coreference']['entities'][cluster_idx] = mention_idxs       
      for cluster_idx, mention_ids in enumerate(self.event_clusters[idx]):
        mention_idxs = [eventid2idx[m_id] for m_id in mention_ids]
        sen_obj['coreference']['events'][cluster_idx] = mention_idxs

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
      example_list = generate_json(image_id, ann_file, example_list)
    return example_list 

def download_video(m2e2_caption):
    m2e2_image_caption = json.load(open(m2e2_caption))
    for _, caption_dict in m2e2_image_caption.items():
      youtube_id = caption_dict['id'].split('v=')[-1]
      video_name = os.path.join(img_dir, youtube_id+'.mp4')
      if not os.path.isfile(video_name):
        os.system('youtube-dl -i -f mp4 -o {} {}'.format(video_name, caption_dict['id']))

def main(grounding_dir, img_dir, m2e2_caption, m2e2_annotation_dir, out_prefix='m2e2'):
    '''
    :param grounding_dir: str, directory name for the grounding meta info files
    :param m2e2_annotation_dir: str, json file name storing the dictionary with the format:
                             [image id]: {[caption id]: 'cap': [caption text], 'url': [url to the image]}
    '''
    m2e2_image_caption = json.load(open(m2e2_caption))
    pairs = list()
    count = 0
    
		# Create a list of (youtube_id, ann_file); download images/videos to img_dir
    download_video(m2e2_caption) 

    ''' XXX 
		# Filter out unannotated examples
    ann_files = []
    for ann_file in os.listdir(os.path.join(m2e2_annotation_dir, '*.ann')):
      with open(os.path.join(m2e2_annotation_dir, ann_file), 'r') as f:
        if len(f.readlines()) > 0:
          ann_files.append(ann_file)
    keys = sorted(m2e2_image_caption) 
    for ann_file in ann_files:
      caption_dict = m2e2_image_caption[keys[int(ann_file.split('.')[0])]] 
      youtube_id = caption_dict['id'].split('v=')[-1]
      if not os.path.isfile(os.path.join(img_dir, youtube_id+'.mp4')):
        os.system('youtube-dl -f mp4 -i {}'.format(caption_dict['id']))
      pairs.append((youtube_id, ann_file))

    result = generate_json_all(pairs)
    _file = codecs.open(os.path.join(grounding_dir, out_prefix + '.json'))  
    json.dump(result, _file, indent=2)
    '''

if __name__ == '__main__':
  grounding_dir = ''
  img_dir = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/MultimediaEventCoreference/m2e2/data/video_m2e2/videos/'
  m2e2_caption = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/MultimediaEventCoreference/m2e2/data/video_m2e2/video_m2e2.json'
  m2e2_annotation_dir = ''
  main(grounding_dir, img_dir, m2e2_caption, m2e2_annotation_dir)
