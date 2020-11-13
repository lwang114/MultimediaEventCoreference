# -*- encode:utf-8; tab-width:2 -*-
# from stanfordcorenlp import StanfordCoreNLP
from nltk import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
import json
import codecs
import os
from collections import defaultdict
import pickle
import random
import math
import spacy
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

nlp = spacy.load('en_core_web_sm')
dep_parser = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz')
# dep_parser._model = dep_parser._model.cuda() 

# dep_parser = CoreNLPDependencyParser(url='http://localhost:9000', tagtype='ner')
# ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
sent_tokenizer = PunktSentenceTokenizer()

def generate_json(img_id, caption, example_list):
    '''
    :param img_id: str, image id of the current example
    :param caption: str, caption text of the current example
    :param example_list: list of processed examples of the format
                         {'image': str, image id,
                          'sentence_id': str, sentence id,
                          'sentence_start': int, start char position,
                          'sentence_end': int, end char position,
                          'sentence': str, caption text,
                          'words': list, word tokens,
                          'index': list of tuple of (start char position, end char position),
                          'pos-tags': list of str of pos tags,
                          'golden-entity-mentions': list of entity mentions of [start, end]
                          'stanford-colcc': list of str of dependency parsing results '[deprel]/dep=[word idx]/gov=[head idx]'}
    :return example_list: updated list of processed examples
    '''
    sentences = sent_tokenizer.span_tokenize(caption)
    if sum(1 for _ in sentences) > 1:
        # if caption contain multiple sentence, ignore this datapoint
        # print('multiple sentences: ', caption)
        return example_list

    sent_idx = 0
    for sentence_start, sentence_end in sent_tokenizer.span_tokenize(caption):
        # each sentence
        sen_obj = dict()
        sen_obj["image"] = img_id
        sen_obj["sentence_id"] = '%s__%d' % (img_id, sent_idx)
        sen_obj["sentence_start"] = sentence_start
        sen_obj["sentence_end"] = sentence_end
        sen_obj['sentence'] = caption[sen_obj["sentence_start"]:sen_obj["sentence_end"]]
        sen_obj['words'] = list()
        sen_obj['index'] = list()
        sen_obj['pos-tags'] = list()
        sen_obj['stanford-colcc'] = list()
        sen_obj['golden-entity-mentions'] = list()
        sent_idx = sent_idx + 1

        # Tokenize the sentence
        tokens = nlp(caption, disable=['parser', 'textcat'])        
        char_pos = 0
        for token in tokens:
          sen_obj['words'].append(token.text)
          end_pos = char_pos + len(token.text) - 1
          sen_obj['index'].append([char_pos, end_pos])
          char_pos = end_pos + 1
          if token.whitespace_:
            char_pos += 1
           
          # Predict pos-tag
          sen_obj['pos-tags'].append(token.tag_)        
        # NER
        ner_info = word2entity(tokens)
        # print('ner_info', ner_info)
        for entity_id in ner_info:
          entity_obj = dict()
          entity_obj['entity-type'] = ner_info[entity_id]['type']
          entity_obj['text'] = ' '.join(ner_info[entity_id]['words'])
          entity_obj['start'] = ner_info[entity_id]['start']
          entity_obj['end'] = ner_info[entity_id]['end']
          sen_obj['golden-entity-mentions'].append(entity_obj)


        # Dependency parsing
        instance = dep_parser._dataset_reader.text_to_instance(sen_obj['words'], sen_obj['pos-tags'])
        parse = dep_parser.predict_instance(instance)
        for word_idx, (head, deprel) in enumerate(zip(parse['predicted_heads'], parse['predicted_dependencies'])):
          sen_obj['stanford-colcc'].append('{}/dep={}/gov={}'.format(deprel, word_idx, head-1))

        example_list.append(sen_obj)
        # print('sen_obj', sen_obj)
        sent_idx += 1
    return example_list

def generate_json_all(pair_list):
    example_list = list()
    for image_id, caption in pair_list:
        # print(image_id, caption)
        # try:
        example_list = generate_json(image_id, caption, example_list)
        # except:
        #     print('[ERROR] run Stanford CoreNLP error', image_id, caption)
        # print(example_list)

    return example_list

def word2entity(tokens):
    '''
    combine the adjacent words
    :param tokens: list of SpaCy token objects
    :return ner_info: dict in format:
                      {entity_id: {'indexes': list of token indices, 
                                   'words': list of word tokens,
                                   'type': ner tag}} 
    '''
    ners = [token.ent_type_ for token in tokens]
    words = [token.text for token in tokens]
    iobs = [token.ent_iob_ for token in tokens]
    
    word_num = len(ners)
    ner_info = defaultdict(lambda : defaultdict()) # entity_id -> entities
    entity_id = -1
    for idx in range(word_num):
        ner_tag = ners[idx]
        iob_tag = iobs[idx]
        if iob_tag == 'O':
          continue
        elif iob_tag == 'B':
          # the beginning of an entity
          entity_id += 1
          ner_info[entity_id]['indexes'] = list()
          ner_info[entity_id]['words'] = list()
          ner_info[entity_id]['type'] = ner_tag
        ner_info[entity_id]['indexes'].append(idx)
        ner_info[entity_id]['words'].append(words[idx])

    # add start/end offset
    for entity_id in ner_info:
        # using index
        ner_info[entity_id]['start'] = ner_info[entity_id]['indexes'][0]
        ner_info[entity_id]['end'] = ner_info[entity_id]['indexes'][-1] + 1
    
    return ner_info

def test(grounding_dir):
    _file = codecs.open(os.path.join(grounding_dir, "grounding_test.json"), 'w', 'utf-8')

    # image = 'IMAGE_2017.jpg'
    # caption = 'This is Barack Obama, who is the president of the U.S.'
    # result = generate_json(image, caption)

    test_list = [('IMAGE_2017.jpg', 'This is Barack Obama, who is the president of the U.S.'),
                 ('IMAGE_2018.jpg', 'This is Lucy Liu Li.')]
    result = generate_json_all(test_list)

    json.dump(result, _file, indent=2)
    # print(result)

# # Split a dataset into a train and test set
# def train_test_split(dataset, split=0.60):
#     train = list()
#     train_size = split * len(dataset)
#     dataset_copy = list(dataset)
#     while len(train) < train_size:
#         index = randrange(len(dataset_copy))
#         train.append(dataset_copy.pop(index))
#     return train, dataset_copy

def download_images(img_dir, voa_caption_full, voa_object_detection, max_num):
  ''' Download the images '''
  voa_image_caption = json.loads(open(voa_caption_full).read())
  # data = pickle.load(open(voa_object_detection, 'rb'))

  count = 0
  for docid in voa_image_caption:
    if count >= max_num:
      break

    for idx in voa_image_caption[docid]:
      url = voa_image_caption[docid][idx]['url']
      suffix = url.split('.')[-1]
      imageID = '%s_%s.%s' % (docid, idx, suffix)
      # if imageID not in data:
      #   continue
      count += 1
      print('Downloading {} from {}'.format(imageID, url))
      os.system('wget {} && mv {} {}/{}'.format(url, url.split('/')[-1], img_dir, imageID))

def main(grounding_dir, voa_caption_full, voa_object_detection, max_num, train_ratio, valid_ratio, test_ratio):
    '''
    :param grounding_dir: str, directory name for the grounding meta info files
    :param voa_caption_full: str, json file name storing the dictionary with the format:
                             [image id]: {[caption id]: 'cap': [caption text], 'url': [url to the image]}
    :param voa_object_detection: str, pickle file storing the bounding box annotation with the format 
                                 [image id]: bbox_info (used only to check if an image has annotation or not)
    :param max_num: int, maximum number of images
    :param train_ratio: float, ratio of training data
    :param valid_ratio: float, ratio of validation data
    :param test_ratio: float, ratio of test data
    '''
    voa_image_caption = json.loads(open(voa_caption_full).read())
    data = pickle.load(open(voa_object_detection, 'rb'))

    pairs = list()
    count = 0
    for docid in voa_image_caption:
        if count >= max_num:
            break
        
        for idx in voa_image_caption[docid]:
            suffix = voa_image_caption[docid][idx]['url'].split('.')[-1]
            imageID = '%s_%s.%s' % (docid, idx, suffix) #'VOA_EN_NW_2012.10.22.1531043_0.jpg'
            if imageID not in data:                
              continue
            count += 1    
            caption = voa_image_caption[docid][idx]['cap']
            pairs.append( (imageID, caption) )
    data_len = len(pairs)
    print('data_len', data_len)

    random.shuffle(pairs)

    train_boundary = math.ceil(train_ratio*data_len)
    test_boundary = math.ceil((train_ratio + test_ratio)*data_len)
    train_data = pairs[:train_boundary]
    test_data = pairs[train_boundary:test_boundary]
    valid_data = pairs[test_boundary:]
    print('train_data', len(train_data))
    print('test_data', len(test_data))
    print('valid_data', len(valid_data))

    _file = codecs.open(os.path.join(grounding_dir, "grounding_train_10000.json"), 'w', 'utf-8')
    result = generate_json_all(train_data)
    # print('train', len(result))
    json.dump(result, _file, indent=2)
    _file = codecs.open(os.path.join(grounding_dir, "grounding_test_10000.json"), 'w', 'utf-8')
    result = generate_json_all(test_data)
    # print('test', result)
    json.dump(result, _file, indent=2)
    _file = codecs.open(os.path.join(grounding_dir, "grounding_valid_10000.json"), 'w', 'utf-8')
    result = generate_json_all(valid_data)
    # print('valid', result)
    json.dump(result, _file, indent=2)

if __name__ == "__main__":
    data_dir = '../../../data'
    img_dir = os.path.join(data_dir, 'voa/rawdata/img/')
    grounding_dir = os.path.join(data_dir, 'grounding')
    voa_caption_full = os.path.join(data_dir, 'voa/rawdata/voa_img_dataset.json')
    voa_object_detection = os.path.join(data_dir, 'voa/object_detect/det_results_voa_oi_1.pkl')
    if not os.path.isdir(img_dir):
      os.mkdir(img_dir)
    if not os.path.isdir(grounding_dir):
      os.mkdir(grounding_dir)

    train_ratio = 0.6
    valid_ratio = 0.2
    test_ratio = 0.2
    max_num = 10
    download_images(img_dir, voa_caption_full, voa_object_detection, max_num)
    main(grounding_dir, voa_caption_full, voa_object_detection, max_num, train_ratio, valid_ratio, test_ratio)
