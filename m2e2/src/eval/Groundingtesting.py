import os
from math import ceil

import torch
from torchtext.data import BucketIterator
from torch.nn import functional as F
from PIL import Image
import numpy as np
import json

import sys
#sys.path.append('/dvmm-filer2/users/manling/mm-event-graph2')
from src.util.util_model import progressbar
from src.dataflow.numpy.data_loader_grounding import unpack_grounding


class GroundingTester():
    def __init__(self):
        pass

    def calculate_lists(self, y, y_):
        '''
        for a sequence, whether the prediction is correct
        note that len(y) == len(y_)
        :param y:
        :param y_:
        :return:
        '''
        ct = 0
        p2 = len(y_)
        p1 = len(y)
        for i in range(p2):
            if y[i] == y_[i]:
                ct = ct + 1
        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1

    def calculate_sets_no_order(self, y, y_):
        '''
        for each predicted item, whether it is in the gt
        :param y: [batch, items]
        :param y_: [batch, items]
        :return:
        '''
        ct, p1, p2 = 0, 0, 0
        for batch, batch_ in zip(y, y_):
            value_set = set(batch)
            value_set_ = set(batch_)
            p1 += len(value_set)
            p2 += len(value_set_)

            for value_ in value_set_:
                # if value_ == '(0,0,0)':
                if value_ in value_set:
                    ct += 1

        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1

    def calculate_sets_noun(self, y, y_):
        '''
        for each ground truth entity, whether it is in the predicted entities
        :param y: [batch, role_num, multiple_args]
        :param y_: [batch, role_num, multiple_entities]
        :return:
        '''
        # print('y', y)
        # print('y_', y_)
        ct, p1, p2 = 0, 0, 0
        # for batch_idx, batch_idx_ in zip(y, y_):
        #     batch = y[batch_idx]
        #     batch_ = y_[batch_idx_]
        for batch, batch_ in zip(y, y_):
            # print('batch', batch)
            # print('batch_', batch_)
            p1 += len(batch)
            p2 += len(batch_)
            for role in batch:
                found = False
                entities = batch[role]
                for entity in entities:
                    for role_ in batch_:
                        entities_ = batch_[role_]
                        if entity in entities_:
                            ct += 1
                            found = True
                            break
                    if found:
                        break

        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1

    def calculate_sets_triple(self, y, y_):
        '''
        for each role, whether the predicted entities have overlap with the gt entities
        :param y: dict, role -> entities
        :param y_: dict, role -> entities
        :return:
        '''
        ct, p1, p2 = 0, 0, 0
        # for batch_idx, batch_idx_ in zip(y, y_):
        #     batch = y[batch_idx]
        #     batch_ = y_[batch_idx_]
        for batch, batch_ in zip(y, y_):
            p1 += len(batch)
            p2 += len(batch_)
            for role in batch:
                entities = batch[role]
                if role in batch_:
                    entities_ = batch_[role]
                    for entity_ in entities_:
                        if entity_ in entities:
                            ct += 1
                            break

        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1

def calculate_auc(S): # TODO
  pass

def calculate_recall(S):
  '''
  :param S: n x n FloatTensor of similarity score between each image and caption
  :return W_r1, W_r5, W_r10: caption recall@1, 5, 10
  :return I_r1, I_r5, I_r10: image recall@1, 5, 10
  '''
  I2W_scores, I2W_ind = S.topk(10, 0)
  W2I_scores, W2I_ind = S.topk(10, 1)
  n = S.size(0)
  W_r1, W_r5, W_r10 = 0., 0., 0.
  I_r1, I_r5, I_r10 = 0., 0., 0.
  for i in range(n):
    W_foundind = -1
    I_foundind = -1
    for ind in range(10):
      if I2W_ind[ind, i] == i:
        W_foundind = ind
      if W2I_ind[i, ind] == i:
        I_foundind = ind
    
    if W_foundind == 0:
      W_r1 += 1.
    if I_foundind == 0:
      I_r1 += 1.    

    if 0 <= W_foundind < 5:
      W_r5 += 1.
    if 0 <= I_foundind < 5:
      I_r5 += 1.  

    if 0 <= W_foundind < 10:
      W_r10 += 1.
    if 0 <= I_foundind < 10:
      I_r10 += 1.
    
  W_r1 /= n
  W_r5 /= n
  W_r10 /= n
  I_r1 /= n
  I_r5 /= n
  I_r10 /= n
  print('Caption Recall@1={:.3f}\tRecall@5={:.3f}\tRecall@10={:.3f}'.format(W_r1, W_r5, W_r10))
  print('Image Recall@1={:.3f}\tRecall@5={:.3f}\tRecall@10={:.3f}'.format(I_r1, I_r5, I_r10))

def compute_similarity_matrix(bbox_embeddings, image_embeddings, word_embeddings, sentence_embeddings):
  # Compute the similarity matrix between each image region and caption word  
  if bbox_embeddings.ndim == 2:
    bbox_embeddings = bbox_embeddings.unsqueeze(1) 
  S_entity = torch.matmul(bbox_embeddings, word_embeddings.permute(0, 2, 1))

  # Compute the similarity scores for the current batch
  S_event = torch.mm(image_embeddings, sentence_embeddings.t())
  return S_entity, S_event


def batch_process_grounding(batch_unpacked, model, tester):
    words, x_len, postags, entitylabels, adjm, \
    image_id, image, bbox_entities_id, bbox_entities_region, bbox_entities_label, object_num_batch, \
    sent_id, entities = batch_unpacked
    print('image.size(): {}'.format(image.size()))

    BATCH_SIZE = image.size(0)
    
    if bbox_entities_region is None:
      OBJECT_LEN = model.sr_model.role_num
      SEQ_LEN = OBJECT_LEN + 1
    else:		
      OBJECT_LEN = bbox_entities_region.size()[-4]
      SEQ_LEN = OBJECT_LEN + 1 

    # Compute the common embedding for each word and for each region
    verb_emb_common, noun_emb_common, verb_emb, noun_emb, heatmap = \
      model.sr_model.get_common_feature(image_id, image, bbox_entities_id, 
                                        bbox_entities_region, bbox_entities_label, 
                                        object_num_batch, BATCH_SIZE, OBJECT_LEN, SEQ_LEN) 
    word_common, word_mask, word_emb = model.ed_model.get_common_feature(words, x_len, postags, entitylabels, adjm)

    # Compute the common space embeddings for the batch
    image_common, sent_common, word2noun_att_output, word_common, noun2word_att_output, noun_emb_common = \
        model.similarity(verb_emb_common, noun_emb_common, verb_emb, noun_emb, word_common, word_emb, word_mask)
    print('image_common.size(): {}, sent_common.size(): {}'.format(image_common.size(), sent_common.size())) # XXX
    
    image_info = {'image_id': image_id, 
                  'bbox_entities_label': bbox_entities_label, 
                  'bbox_entities_id': bbox_entities_id}
    print('bbox entities id: {}'.format(bbox_entities_id)) # XXX
    print('bbox_entities label: {}'.format(bbox_entities_label))
    return verb_emb_common, image_common, word_common, sent_common, image_info, entitylabels


def run_over_batch_grounding(batch, model, tester, ee_hyps,
                             img_dir, transform, add_object=False,
                             object_results=None, object_label=None,
                             object_detection_threshold=.2, vocab_objlabel=None):

    # XXX try:
    #    # words, x_len, postags, entitylabels, adjm, image_id, image = unpack_grounding(batch, device, transform,
    #    #                                                                           img_dir, ee_hyps)
    batch_unpacked = unpack_grounding(batch, 'cpu', transform, img_dir, ee_hyps,
                                      load_object=add_object, object_results=object_results,
                                      object_label=object_label,
                                      object_detection_threshold=object_detection_threshold,
                                      vocab_objlabel=vocab_objlabel)
    # XXX except:
    #    # if the batch is a bad batch, Nothing changed, return directly
    #    return running_loss, cnt, all_captions, all_captions_, all_images, all_images_

    if batch_unpacked is None:
        # if the batch is a bad batch, Nothing changed, return directly
        return emb_bbox, emb_image, emb_word, emb_sentence 

    emb_bbox, emb_image, emb_word, emb_sentence, image_info, entitylabels\
        = batch_process_grounding(batch_unpacked,
                                  model, tester)

    return emb_bbox, emb_image, emb_word, emb_sentence, image_info, entitylabels 


def run_over_data_grounding(model, data_iter, tester, ee_hyps,
                            img_dir, transform, add_object=False,
                            object_results=None, object_label=None,
                            object_detection_threshold=.2, vocab_objlabel=None,
                            out_dir='./'):
    model.eval()
    bbox_embeddings = []
    image_embeddings = []
    word_embeddings = []
    sentence_embeddings = []
    image_dicts = {'image_id': [], 'bbox_entities_label': [], 'bbox_entities_id': []}
    entitylabels = [] 

    # print(data_iter)
    for batch in data_iter:
      emb_bbox, emb_image, emb_word, emb_sentence, image_info, ent_labels\
          = run_over_batch_grounding(batch, 
                                     model, tester, ee_hyps,
																		 img_dir, transform, add_object=add_object,
                                     object_results=object_results,
                                     object_label=object_label,
                                     object_detection_threshold=object_detection_threshold,
                                     vocab_objlabel=vocab_objlabel
                                    )
      image_dicts['image_id'].extend(image_info['image_id'])
      if add_object:
        image_dicts['bbox_entities_label'].extend(image_info['bbox_entities_label'].data.cpu().numpy().tolist())
        image_dicts['bbox_entities_id'].extend(image_info['bbox_entities_id'])      
      entitylabels.extend(ent_labels)
      bbox_embeddings.append(emb_bbox)
      image_embeddings.append(emb_image)
      word_embeddings.append(emb_word)
      sentence_embeddings.append(emb_sentence)
     
    # Compute similarity matrix
    bbox_embeddings = torch.cat(bbox_embeddings)
    image_embeddings = torch.cat(image_embeddings)
    word_embeddings = torch.cat(word_embeddings)
    sentence_embeddings = torch.cat(sentence_embeddings)

    S_entity, S_event = compute_similarity_matrix(bbox_embeddings, image_embeddings, word_embeddings, sentence_embeddings)
    calculate_recall(S_event)

    # Save the similarity matrix
    np.save(os.path.join(out_dir, 'event_similarity_matrix.npy'), S_event.data.cpu().numpy())
    S_entity_dict = [{'image_id': img_id, 
                      'bbox_entities_id': entity_id, 
                      'bbox_entities_label': bbox_entity_label, 
                      'entity_labels': entitylabel, 
                      'scores': S_entity_i}\
                      for img_id, entity_id, bbox_entity_label, entitylabel, S_entity_i in \
                          zip(image_dicts['image_id'], image_dicts['bbox_entities_label'],\
                              image_dicts['bbox_entities_id'], entitylabels, S_entity.data.cpu().numpy().tolist())]
    json.dump(S_entity_dict, open(os.path.join(out_dir, 'entity_similarity_matrix.json'), 'w'), indent=2, sort_keys=True) 


def grounding_test(model, test_set, 
            tester, parser, other_testsets, transform, vocab_objlabel=None):
  test_iter = BucketIterator(test_set, batch_size=parser.batch,
                             train=False, shuffle=False, device=-1,
                             sort_key=lambda x: len(x.POSTAGS))  
  object_results, object_label, object_detection_threshold = test_set.get_object_results()
  run_over_data_grounding(
      data_iter=test_iter,
      model=model,
      tester=tester,
			ee_hyps=parser.ee_hps,
      img_dir=parser.img_dir,
      transform=transform,
      add_object=parser.add_object,
      object_results=object_results,
      object_label=object_label,
      object_detection_threshold=object_detection_threshold,
      vocab_objlabel=vocab_objlabel,
      out_dir=parser.out
	 ) 
