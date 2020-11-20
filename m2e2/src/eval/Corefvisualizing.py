import json
from collections import defaultdict
import numpy as np
import os
import shutil
import pickle as pkl
import argparse

class CorefVisualizer:
  def __init__(self, entity_score_json, event_score_npy, 
               entity_mapping_file, noun_mapping_file,
               grounding_file):
    '''
    :param entity_score_json: str, name of the json file of entity coref score info of format 
           {'image_id': str of image id,
            'bbox_entities_label': list of integer of bbox entity labels,
            'bbox_entities_id': list of str of bbox entity ids,
            'entity_labels': list of integer of text entity labels,
            'scores': list of list of floats, i-th row, j-th column
                      stores the alignment score between i-th region 
                      and j-th word}
    :param event_score_npy: str, name of the npy file of event coref scores 
    '''
    self.event_scores = np.load(event_score_npy)
    grounding_dicts = json.load(open(grounding_file))
    self.noun_id2s = self.load_object_category_mapping(noun_mapping_file)
    entity_score_dicts = json.load(open(entity_score_json))
    
    self.img_ids = [entity_score_dict['image_id'] for entity_score_dict in entity_score_dicts]
    self.ret_img_ids = []
    self.y_bbox = []
    self.y_bbox_ = []
    self.captions = []

    for ex, entity_score_dict in enumerate(entity_score_dicts):
      # Extract retrieved image ids      
      ret_img_ids = [self.img_ids[idx] for idx in np.argsort(-self.event_scores[ex])[:10].tolist()] 
      self.ret_img_ids.append(ret_img_ids)

      # Extract caption
      words = None
      for ground_dict in grounding_dicts:
        if self.img_ids[ex] == ground_dict['image']: 
          words = ground_dict['words']
          break
      # print(self.img_ids[ex])
      self.captions.append(' '.join(words))

      if len(entity_score_dict['bbox_entities_label']) > 0:
        # Extract ground truth bbox entity type names
        bbox_label = entity_score_dict['bbox_entities_label'] 
        y_bbox = [self.noun_id2s[int(noun_idx)] for noun_idx in bbox_label]
        self.y_bbox.append(y_bbox)
      
        # Extract predicted bbox entity type names     
        max_box_idxs = np.argmax(np.asarray(entity_score_dict['scores']), axis=0).tolist() 
        
        # Find the map from bounding box to entities aligned to the box
        y_bbox_ = defaultdict(list)
        for box_idx, token in zip(max_box_idxs, words):
          # parts = entity_score_dict['bbox_entities_id'][box_idx].split('_')
          # x1 = int(parts[0])
          # y1 = int(parts[1]) 
          # x2 = int(parts[2])
          # y2 = int(parts[3])
          y_bbox_[box_idx].append(token)
        self.y_bbox_.append(y_bbox_)
      else:
        self.y_bbox.append([])
        self.y_bbox_.append([])
        
  def load_object_category_mapping(self, noun_mapping_file):
    '''
    :param noun_mapping_file: str, name of .txt file with each row [noun id],[str token],[is boxable]
    :return noun_id2s: dict, noun id -> str token
    '''
    noun_id2s = pkl.load(open(noun_mapping_file, 'rb'))
    # with open(noun_mapping_file, 'r') as f:
    #   for line in f:
    #     noun_id, noun_s = line.split(',')[:2] 
    #     noun_id2s[noun_id] = noun_s
    return noun_id2s
      
    
  def visualize_html(self, visual_path, image_path=None, display_path=None):
    '''
    Visualize the cross-modal entity coreference results.
    :param visual_path: str, path to the output directory
    :param image_path: str, path to the image directory
    :param display_path: str, path printed in the html file
    ''' 
    if not image_path:
        image_path = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/MultimediaEventCoreference/m2e2/data/voa/rawdata/img/'
    if not display_path:
        display_path = visual_path

    if not os.path.isdir(visual_path):
        os.mkdir(visual_path)
        os.mkdir(os.path.join(visual_path, 'img'))

    for ex, img_id in enumerate(self.img_ids):
      f_html = open(os.path.join(visual_path, '{}.html'.format(img_id)), 'w')
      # Show the image
      if not os.path.isfile(os.path.join(visual_path, 'img', img_id)):
        shutil.copyfile(os.path.join(image_path, img_id), os.path.join(visual_path, 'img', img_id))
      
      f_html.write('<img src=\"' + os.path.join(display_path, 'img', img_id) + '\" width=\"300\">\n<br>')     
      # Display the caption for the current image
      f_html.write('[caption] {} \n<br>'.format(self.captions[ex]))

      if len(self.y_bbox[ex]) > 0:
        # Display the correct entity labels for the current image
        f_html.write('[box_entity_ground_truth] {} \n<br>'.format(' '.join(self.y_bbox[ex])))

        # Display the predicted entity labels for the current image
        f_html.write('[box_entity_predicted]')
        for _, ys_ in self.y_bbox_[ex].items():	
          f_html.write('|{}'.format(' '.join(ys_))) 
        f_html.write('\n<br>')

      # Display the retrieved images for the current caption 
      f_html.write('[retrieved_images]\n<br>\n')
      for ret_img_id in self.ret_img_ids[ex]:
        if not os.path.isfile(os.path.join(visual_path, 'img', img_id)):
          shutil.copyfile(os.path.join(image_path, ret_img_id), os.path.join(visual_path, 'img', ret_img_id))

        f_html.write('<img src=\"' + os.path.join(display_path, 'img', ret_img_id) + '\"width=\"300\">\n<br>')
      f_html.write('\n<br>')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', default='../engine/out')
  parser.add_argument('--data_dir', default='../../data')
  
  args = parser.parse_args()  
  exp_dir = args.exp_dir
  data_dir = args.data_dir
  visualizer = CorefVisualizer(entity_score_json=os.path.join(exp_dir, 'entity_similarity_matrix.json'),
                               event_score_npy=os.path.join(exp_dir, 'event_similarity_matrix.npy'),
                               entity_mapping_file=os.path.join(exp_dir, 'entity.vec'), 
                               noun_mapping_file=os.path.join(data_dir, 'vocab/vocab_situation_noun.pkl'), 
                               grounding_file=os.path.join(data_dir, 'grounding/backup/11_19_2020/grounding_test_10000.json')) # XXX
  visualizer.visualize_html(visual_path=os.path.join(exp_dir, 'visualization'),
                            image_path=os.path.join(data_dir, 'voa/rawdata/img'),
                            display_path='/Users/liming/research/PhD_research/fall2020/pictures/visualization')
