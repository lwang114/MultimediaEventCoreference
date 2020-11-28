# -*- coding:utf-8 -*-
import json
import os
import pickle
import codecs
import nltk

dep_parser = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz')
def generate_json(img_id, text, anns, example_list):
    '''
    :param img_id: str, image id of the current example
    :param anns: list of dict, annotations of sentences of the example with format:
                {'doc_id': str, 
                 'sent_id': str, [doc_id]_[sentence number],
                 'token_ids': str, '%(doc_id):%(start char)-%(end char)',
                 'tokens': list of str,
                 'graph': {'entities': list of list of 
                                       [start token, end token, 
                                        type, mention type, score],
                           'triggers': list of list of
                                       [start token, end token, type, score],
                           'roles': list of list of 
                                    [start token, end token, type, score]}}
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
                                 'text': str},
                     'event-type': str,
                     'arguments': list of {'role': str, 'start': int, 'end': int, 'text': str},
                'stanford-colcc': list of str of dependency parsing results 
                                  '[deprel]/dep=[word idx]/gov=[head idx]',
                'coreference': {'entities': list of list of ints, 'events': list of list of ints},
                'mention_ids': {'entities': list of str,
                                'events': list of str}}               
    '''
    char_pos = 0
    for sent_idx, ann in enumerate(anns):
      sen_obj = dict()
      sent_id = ann['sent_id']
      sent_start, sent_end = ann['token_ids'][0].split(':')[-1].split('-')
      index = [[int(pos) for pos in token_id.split(':')[-1].split('-')]\
                for token_id in ann['token_ids']]
      tokens = ann['tokens']
      
      # Tokenization
      sen_obj['image'] = img_id
      sen_obj['sentence_id'] = sent_id
      sen_obj['sentence_start'] = int(sent_start)
      sen_obj['sentence_end'] = int(sent_end)
      sen_obj['sentence'] = text
      sen_obj['words'] = tokens
      sen_obj['index'] = index
      sen_obj['stanford-colcc'] = list() 
      sen_obj['golden-entity-mentions'] = list()
      sen_obj['golden-event-mentions'] = list()

      # POS tagging
      postags = [t[1] for t in nltk.pos_tag(tokens)]
      sen_obj['pos-tags'] = postags

      # Dependency parsing
      instance = dep_parser._dataset_reader.text_to_instance(sen_obj['words'], sen_obj['pos-tags'])
      parser = dep_parser.predict_instance(instance)
      for word_idx, (head, deprel) in enumerate(zip(parse['predicted_heads'], parse['predicted_dependencies'])): 
        sen_obj['stanford-colcc'].append('{}/dep={}/gov={}'.format(deprel, word_idx, head-1))

      # Entities
      for ne in ann['graph']['entities']:
        sen_obj['golden-entity-mentions'].append({
          'start': ne[0],
          'end': ne[1],
          'entity-type': ne[2],
          'text': ' '.join(tokens[ne[0]:ne[1]+1])})
      
      # Events
      for trigger in ann['graph']['triggers']:
        sen_obj['golden-event-mentions'].append({
          'start': trigger[0],
          'end': trigger[1],
          'event-type': trigger[2],
          'text': ' '.join(tokens[trigger[0]:trigger[1]+1])})
      example_list.append(sen_obj)
    return example_list
        

def main(grounding_dir, video_dir, m2e2_caption, 
         m2e2_annotation_dir, out_prefix='video_m2e2'):
    '''
    :param grounding_dir: str, directory of the meta info file
    :param video_dir: str, directory of the videos
    :param m2e2_caption: str, json file name storing the dictionary with the format:
                         [short desc]: {'id': [caption text], 
                                        'long_desc': [url to the image]}
    :param m2e2_annotation_dir: str, directory name of the annotations
    '''
    m2e2_image_caption = json.load(open(m2e2_caption))
    result = list()
    count = 0

    keys = sorted(m2e2_image_caption)
    for k in keys:    
      youtube_id = m2e2_image_caption[k]['id']
      text = m2e2_image_caption[k]['long_desc']
      ann_file = os.path.join(m2e2_annotation_dir, youtube_id+'.json')
      anns = json.loads(codecs.open(ann_file).read())
      result = generate_json(img_id, text, anns, result)

    with codecs.open(os.path.join(grounding_dir, out_prefix+'.json'), 'r'):
      json.dump(result, f, indent=2)

if __name__ == '__main__':
  data_dir = '../../../data'
  grounding_dir = os.path.join(data_dir, 'video_m2e2') 
  video_dir = os.path.join(data_dir, 'video_m2e2/videos')
  m2e2_caption = os.path.join(data_dir, 'video_m2e2/video_m2e2.json')
  m2e2_annotation_dir = os.path.join(data_dir, 'video_m2e2/')
  main()
