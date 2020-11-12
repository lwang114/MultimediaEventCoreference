# from stanfordcorenlp import StanfordCoreNLP
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.parse import CoreNLPParser
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

def test_nlp():
    '''
    test Stanford CoreNLP
    :return:
    '''
    content = 'This is a test sentence. This a Lucy Liu.'
    print(sent_tokenize(content))
    for sentence_start, sentence_end in sent_tokenizer.span_tokenize(content):
        sen_obj = dict()
        sen_obj["sentence_start"] = sentence_start
        sen_obj["sentence_end"] = sentence_end
        sen_obj['sentence'] = content[sen_obj["sentence_start"]:sen_obj["sentence_end"]]
        sen_obj['words'] = list()
        sen_obj['index'] = list()

    data = dict()

    data['token'] = []
    data['stanford_pos'] = []
    data['stanford_ner'] = [] #use training data entities instead
    data['stanford_head'] = []
    data['stanford_deprel'] = []

    parse, = dep_parser.raw_parse(content)
    print(parse)
    #    print(parse.to_conll(4))
    for line in parse.to_conll(4).strip().split('\n'):
        #        print(repr(line))
        ele = line.split('\t')
        data['token'].append(ele[0])
        data['stanford_pos'].append(ele[1])
        data['stanford_head'].append(ele[2])
        data['stanford_deprel'].append(ele[3])

    print(dep_parser.tag(data['token']))
    # data['stanford_ner'] =
    # [('Rami', 'PERSON'), ('Eid', 'PERSON'), ('is', 'O'), ('studying', 'O'), ('at', 'O'), ('Stony', 'ORGANIZATION'),
    #  ('Brook', 'ORGANIZATION'), ('University', 'ORGANIZATION'), ('in', 'O'), ('NY', 'STATE_OR_PROVINCE')]
    print(data)

# def generate_source(file_path):
#     source_dict['doc_content']
#     source_dict['entity_dict']
#     source_dict['event_list']

def generate_json(img_id, caption, example_list): # TODO
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
					sen_obj['index'].append([char_pos, end_pos])
					char = end_pos + 1
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


				# TODO Dependency parsing
				instance = dep_parser._dataset_reader.text_to_instance(sen_obj['words'], sen_obj['pos-tags'])
				parse = dep_parser.predict_instance(instance)
				for word_idx, (head, deprel) in enumerate(zip(parse['predicted_heads'], parsed_sent['predicted_dependencies'])):
					sen_obj['stanford-colcc'].append('{}/dep={}/gov={}'.format(deprel, word_idx, head-1))
				'''
				parse, = dep_parser.raw_parse(sen_obj['sentence'])
        pre = 0
        word_idx = 0
        for line in parse.to_conll(4).strip().split('\n'):
            ele = line.split('\t')

            # print(ele)
            word = ele[0].replace('-LRB-', '(').replace('-RRB-', ')')
            sen_obj['words'].append(word)
            bg = sen_obj['sentence'].find(word, pre)
            ed = bg + len(word) - 1
            pre = ed + 1
            # print(len(word), pre)
            # print('\"' + sen_obj['sentence'][bg:ed + 1] + '\"', '\"' + word + '\"')
            # assert sen_obj['sentence'][bg:ed + 1] == word
            if sen_obj['sentence'][bg:ed + 1] != word:
                return example_list
            sen_obj['index'].append((bg + sen_obj['sentence_start'], ed + sen_obj['sentence_start']))

            pos = ele[1]
            sen_obj['pos-tags'].append(pos)

            head = ele[2]
            deprel = ele[3]
            sen_obj['stanford-colcc'].append('%s/dep=%d/gov=%d' % (deprel, word_idx, int(head)-1) )
                # deprel + "/dep=" + str(word_idx) + "/gov=" + str(head-1))
            word_idx = word_idx + 1
				
        ners = dep_parser.tag(sen_obj['words'])
        # print('ners', ners)
        # word_num = len(ners)
        # for idx_i in range(word_num):
        #     ner_tag = ners[idx_i][-1]
        #     word = ners[idx_i][0]
        #     if word == sen_obj['words'][idx_i]:
        #         if ner_tag != 'O':
        #             entity_obj = dict()
        #             entity_obj['entity-type'] = ner_tag
        #             entity_obj['text'] = word
        #             entity_obj['start'] = sen_obj['index'][idx_i][0]
        #             entity_obj['end'] = sen_obj['index'][idx_i][1]
        #             sen_obj['golden-entity-mentions'].append(entity_obj)
        #     else:
        #         print('word', word)
        #         print('word_', sen_obj['words'][idx_i])
				'''
        # sen_obj['golden-event-mentions'] = list()

        example_list.append(sen_obj)
        # print('sen_obj', sen_obj)
        sent_idx += 1
    return example_list

def generate_json_all(pair_list):
    example_list = list()
    for image_id, caption in pair_list:
        # print(image_id, caption)
        try:
            example_list = generate_json(image_id, caption, example_list)
        except:
            print('[ERROR] run Stanford CoreNLP error', image_id, caption)
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
		ners = [token.ner_, for token in tokens]
		words = [token.text for token in tokens]
		iobs = [token.iob_ for token in tokens]
		
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
	data = pickle.load(open(voa_object_detection, 'rb'))

	for docid in voa_image_caption:
		count += 1
		if count > max_num:
			break

		for idx in voa_image_caption[docid]:
			url = voa_image_caption[docid][idx]['url']
			suffix = url.split('.')[-1]
			imageID = '%s_%s.%s' % (docid, idx, suffix)
			if imageID not in data:
				continue
			os.system('wget {} > {}/{}'.format(url, img_dir, imageID))

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
        count += 1
        if count > max_num:
            break

        for idx in voa_image_caption[docid]:
            suffix = voa_image_caption[docid][idx]['url'].split('.')[-1]
            imageID = '%s_%s.%s' % (docid, idx, suffix) #'VOA_EN_NW_2012.10.22.1531043_0.jpg'
            if imageID not in data:
                continue
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
    grounding_dir = '/dvmm-filer2/users/manling/mm-event-graph2/data/grounding'
    voa_caption_full = '/dvmm-filer2/users/manling/mm-event-graph2/data/voa/rawdata/voa_img_dataset.json'
    voa_object_detection = '/dvmm-filer2/users/manling/mm-event-graph2/data/voa_caption/object_detect/det_results_voa_oi_1.pkl'

# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
# -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
# -status_port 9000 -port 9000 -timeout 15000 &
#     test_nlp()
#     test(grounding_dir)

    train_ratio = 0.6
    valid_ratio = 0.2
    test_ratio = 0.2
    max_num = 10000
    main(grounding_dir, voa_caption_full, voa_object_detection, max_num, train_ratio, valid_ratio, test_ratio)
