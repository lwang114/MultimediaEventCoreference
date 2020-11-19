import json
from collections import defaultdict
import numpy as np
import os
import shutil

class CorefVisualizer:
  def __init__(self, entity_score_json, event_score_npy, 
               entity_mapping_file, noun_mapping_file,
               grounding_file):
    '''
    :param entity_score_json: str, name of the json file of entity coref score info of format 
           {'img_id': str of image id,
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
    self.noun_id2s = json.load(open(noun_mapping_file)) # TODO
    entity_score_dicts = json.load(open(entity_score_json))
    
    self.img_ids = [entity_score_dict['img_id'] for entity_score_dict in entity_score_dicts]
    self.ret_img_ids = []
    self.y_bbox = []
    self.y_bbox_ = []
    self.captions = []

    for ex, entity_score_dict in enumerate(entity_score_dicts):
      # Extract retrieved image ids      
      ret_img_ids = [self.img_ids[idx] for idx in np.argsort(self.event_scores[:, ex], axis=0)[:10].tolist()] 
      self.ret_img_ids.append(ret_img_ids)

      # Extract caption    
      caption = grounding_dicts[ex]['sentence']
      self.captions.append(caption)

      if len(entity_score_dict['bbox_entities_label']) > 0:
        # Extract ground truth bbox entity type names
        bbox_label = entity_score_dict['bbox_entities_label'] 
        y_bbox = [self.noun_id2s[noun_idx] for noun_idx in bbox_label]
        self.y_bbox.append(y_bbox)
      
        # Extract predicted bbox entity type names     
        max_box_idxs = np.argmax(self.entity_scores[ex], axis=0).tolist() 
        
        # Find the map from bounding box to entities aligned to the box
        y_bbox_ = defaultdict(list)
        for box_idx, token in zip(max_box_idxs, caption):
          # parts = entity_score_dict['bbox_entities_id'][box_idx].split('_')
          # x1 = int(parts[0])
          # y1 = int(parts[1]) 
          # x2 = int(parts[2])
          # y2 = int(parts[3])
          y_bbox_[box_idx].append(token)
        self.y_bbox_.append(y_bbox_)

  def visualize_html(self, visual_path, image_path=None):
    '''
    Visualize the cross-modal entity coreference results.
    :param visual_path: str, path to the output directory
    :param image_path: str, path to the image directory
    ''' 
    if not image_path:
        image_path = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/MultimediaEventCoreference/m2e2/data/voa/rawdata/img/'
        if not os.path.isdir(visual_path):
          os.mkdir(visual_path)
          os.mkdir(os.path.join(visual_path, 'img'))

    for ex, img_id in enumerate(self.img_ids):
      f_html = open(os.path.join(visual_path, '{}.html'.format(img_id)), 'w')
      # Show the image
			if not os.path.isfile(os.path.join(visual_path, 'img', img_id)):
        shutil.copyfile(os.path.join(image_path, img_id), os.path.join(visual_path, 'img', img_id))
      
      f_html.write('<img src=\"' + os.path.join(visual_path, 'img', img_id) + '\" width=\"300\">\n<br>')     
      # Display the caption for the current image
      f_html.write('[caption] {} \n<br>'.format(self.captions[ex]))

      if len(self.y_bbox[ex]) > 0:
        # Display the correct entity labels for the current image
        f_html.write('[box_entity_ground_truth] {} \n<br>'.format(' '.join(self.y_bbox[ex])))

        # Display the predicted entity labels for the current image
        f_html.write('[box_entity_predicted]')
        for ys_ in self.y_bbox_[ex]:
          f_html.write(' {}'.format(','.join(ys_))) 
        f_html.write('\n<br>')

      # Display the retrieved images for the current caption 
      f_html.write('[retrieved_images]\n<br>\n')
      for ret_img_id in self.ret_img_ids[ex]:
        if not os.path.isfile(os.path.join(visual_path, 'img', img_id)):
          shutil.copyfile(os.path.join(image_path, ret_img_id), os.path.join(visual_path, 'img', ret_img_id))

        f_html.write('<img src=\"' + os.path.join(visual_path, 'img', ret_img_id) + '\"width=\"300\">\n<br>')
      f_html.write('\n<br>')
