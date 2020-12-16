# -*- coding: utf-8 -*-
import json
import numpy as np
import codecs
import pyhocon
import os

def get_mention_doc(caption_file, out_prefix):
    '''
    :param caption_file: str of filename of a .txt file. Each line is of the format [doc id] [caption text]
    :param out_prefix: str of the prefix of the output file storing dict of
    {[doc_id]: list of [sent id, token id, token, is entity/event]}
    '''
    sen_dicts = {}
    with open(caption_file, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.split()
            doc_id = '{}_{}'.format(parts[0], idx)
            print('doc id {}'.format(doc_id))
            caption = parts[1:]                

            sen_dicts[doc_id] = []
            for token_id, token in enumerate(caption):
                sen_dicts[doc_id].append([0, token_id, token, True])
    json.dump(sen_dicts, open('{}.json'.format(out_prefix), 'w'), indent=4)

def extract_glove_embeddings(config, glove_file, dimension=300, out_prefix='glove_embedding'):
    ''' Extract glove embeddings for a sentence
    :param doc_json: json metainfo file in m2e2 format
    :return out_prefix: output embedding for the sentences
    '''
    doc_json = os.path.join(config['data_folder'], 'flickr.json')
    documents = json.load(codecs.open(doc_json, 'r', 'utf-8'))
    print('Number of documents: {}'.format(len(documents)))
    vocab = {'$$$UNK$$$': 0}
    # Compute vocab of the documents
    for doc_id in sorted(documents): # XXX
        tokens = documents[doc_id]
        for token in tokens:
            if not token[2].lower() in vocab:
                # print(token[2].lower())
                vocab[token[2].lower()] = len(vocab)
    print('Vocabulary size: {}'.format(len(vocab)))
                
    embed_matrix = [[0.0] * dimension] 
    vocab_emb = {'$$$UNK$$$': 0} 
    # Load the embeddings
    with codecs.open(glove_file, 'r', 'utf-8') as f:
        for line in f:
            segments = line.strip().split()
            if len(segments) == 0:
                print('Empty line')
                break
            word = ' '.join(segments[:-300])
            if word in vocab:
                # print('Found {}'.format(word))
                embed= [float(x) for x in segments[-300:]]
                embed_matrix.append(embed)
                vocab_emb[word] = len(vocab_emb)
    print('Vocabulary size with embeddings: {}'.format(len(vocab_emb)))
    json.dump(vocab_emb, open(out_prefix+'_vocab.json', 'w'), indent=4, sort_keys=True)
    
    # Convert the documents into embedding sequence
    doc_embeddings = {}
    for idx, doc_id in enumerate(sorted(documents, key=lambda x:int(x.split('_')[-1]))): # XXX
        embed_id = doc_id
        print(embed_id)
        tokens = documents[doc_id]
        doc_embedding = []
        for token in tokens:
            token_id = vocab_emb.get(token[2].lower(), 0)
            doc_embedding.append(embed_matrix[token_id])
        print(doc_id, len(tokens), np.asarray(doc_embedding).shape)
        doc_embeddings[embed_id] = np.asarray(doc_embedding)
    np.savez(out_prefix+'.npz', **doc_embeddings)
    
if __name__ == '__main__':
    config_file = 'configs/config_grounded_mml_flickr.json'
    caption_file = '/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/flickr30k_text_captions.txt'
    glove_file = 'm2e2/data/glove/glove.840B.300d.txt'
    
    config = pyhocon.ConfigFactory.parse_file(config_file)
    if not os.path.isdir(config['data_folder']):
        os.makedirs(config['data_folder'])

    # get_mention_doc(caption_file, os.path.join(config['data_folder'], 'flickr'))
    extract_glove_embeddings(config, glove_file, out_prefix=os.path.join(config['data_folder'], 'flickr_glove_embeddings'))
