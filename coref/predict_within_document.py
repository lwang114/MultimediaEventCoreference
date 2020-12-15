from sklearn.cluster import AgglomerativeClustering
import argparse
import pyhocon
from transformers import AutoTokenizer, AutoModel
from itertools import product
import collections
from tqdm import tqdm

from conll import write_output_file
from models import SpanScorer, SimplePairWiseClassifier, SpanEmbedder
from utils import *
from model_utils import *
import subprocess




def init_models(config, device):
    span_repr = SpanEmbedder(config, device).to(device)
    span_repr.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                      "span_repr_{}".format(config['model_num'])),
                                         map_location=device))
    span_repr.eval()
    span_scorer = SpanScorer(config).to(device)
    span_scorer.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                        "span_scorer_{}".format(config['model_num'])),
                                           map_location=device))
    span_scorer.eval()
    pairwise_scorer = SimplePairWiseClassifier(config).to(device)
    pairwise_scorer.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                           "pairwise_scorer_{}".format(config['model_num'])),
                                              map_location=device))
    pairwise_scorer.eval()

    return span_repr, span_scorer, pairwise_scorer



def is_included(docs, starts, ends, i1, i2):
    doc1, start1, end1 = docs[i1], starts[i1], ends[i1]
    doc2, start2, end2 = docs[i2], starts[i2], ends[i2]

    if doc1 == doc2 and (start1 >= start2 and end1 <= end2):
        return True
    return False


def remove_nested_mentions(cluster_ids, doc_ids, starts, ends):
    # nested_mentions = collections.defaultdict(list)
    # for i, x in range(len(cluster_ids)):
    #     nested_mentions[x].append(i)

    doc_ids = np.asarray(doc_ids)
    starts = np.asarray(starts)
    ends = np.asarray(ends)

    new_cluster_ids, new_docs_ids, new_starts, new_ends = [], [], [], []

    for cluster, idx in cluster_ids.items():
        docs = doc_ids[idx]
        start = starts[idx]
        end = ends[idx]


        for i in range(len(idx)):
            indicator = [is_included(docs, start, end, i, j) for j in range(len(idx))]
            if sum(indicator) > 1:
                continue

            new_cluster_ids.append(cluster)
            new_docs_ids.append(docs[i])
            new_starts.append(start[i])
            new_ends.append(end[i])


    clusters = collections.defaultdict(list)
    for i, cluster_id in enumerate(new_cluster_ids):
        clusters[cluster_id].append(i)

    return clusters, new_docs_ids, new_starts, new_ends





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_clustering.json')
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)
    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    create_folder(config['save_path'])
    device = 'cuda:{}'.format(config['gpu_num'][0]) if torch.cuda.is_available() else 'cpu'

    '''
    # Load models and init clustering
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    config['bert_hidden_size'] = bert_model.config.hidden_size
    span_repr, span_scorer, pairwise_scorer = init_models(config, device)
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=config['threshold'])


    # Load data
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    data = create_corpus(config, bert_tokenizer, config.split, is_training=False)

    doc_ids, sentence_ids, starts, ends = [], [], [], []
    all_topic_predicted_clusters = []
    max_cluster_id = 0

    # Go through each topic

    for topic_num, topic in enumerate(data.topic_list):
        print('Processing topic {}'.format(topic))
        total = len(data.topics_list_of_docs[topic_num])
        nbatches = total // config['batch_size'] + 1
        for b in range(nbatches):
          start_idx = b * config['batch_size']
          end_idx = min((b + 1) * config['batch_size'], total)
          batch_idxs = list(range(start_idx, end_idx)) 
          bert_tokens = [data.topics_bert_tokens[topic_num][i] for i in batch_idxs]
          docs_embeddings, docs_length = pad_and_read_bert(bert_tokens, bert_model)
          span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
              config, data, topic_num, docs_embeddings, docs_length, batch_idxs=batch_idxs)

          doc_id, sentence_id, start, end = span_meta_data
          start_end_embeddings, continuous_embeddings, width = span_embeddings
          width = width.to(device)

          labels = data.get_candidate_labels(doc_id, start, end)

          if config['use_gold_mentions']:
              span_indices = labels.nonzero().squeeze(1)
          else:
              k = int(config['top_k'] * num_of_tokens)
              with torch.no_grad():
                  span_emb = span_repr(start_end_embeddings, continuous_embeddings, width)
                  span_scores = span_scorer(span_emb)
              _, span_indices = torch.topk(span_scores.squeeze(1), k, sorted=False)
              # span_indices, _ = torch.sort(span_indices)


          number_of_mentions = len(span_indices)
          start_end_embeddings = start_end_embeddings[span_indices]
          continuous_embeddings = [continuous_embeddings[i] for i in span_indices]
          width = width[span_indices]
          torch.cuda.empty_cache()

          # Prepare all the pairs for the distance matrix
          first, second = zip(*list(product(range(len(span_indices)), repeat=2)))
          first = torch.tensor(first)
          second = torch.tensor(second)

          torch.cuda.empty_cache()
          all_scores = []
          with torch.no_grad():
              # g1 = span_repr(start_end_embeddings[first],
              #                [continuous_embeddings[k] for k in first],
              #                width[first])
              # g2 = span_repr(start_end_embeddings[second],
              #                [continuous_embeddings[k] for k in second],
              #                width[second])
              #
              # scores = pairwise_scorer(g1, g2)
              # scores = torch.sigmoid(scores)
              # all_scores.csv.extend(scores.squeeze(1))
              # torch.cuda.empty_cache()

              for i in range(0, len(first), 10000):
                  # end_max = min(i+100000, len(first))
                  end_max = i+10000
                  first_idx, second_idx = first[i:end_max], second[i:end_max]
                  g1 = span_repr(start_end_embeddings[first_idx],
                                 [continuous_embeddings[k] for k in first_idx],
                                 width[first_idx])
                  g2 = span_repr(start_end_embeddings[second_idx],
                                 [continuous_embeddings[k] for k in second_idx],
                                 width[second_idx])
                  scores = pairwise_scorer(g1, g2)

                  torch.cuda.empty_cache()
                  if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']:
                      g1_score = span_scorer(g1)
                      g2_score = span_scorer(g2)
                      scores += g1_score + g2_score

                  scores = torch.sigmoid(scores)
                  all_scores.extend(scores.detach().cpu().squeeze(1))
                  torch.cuda.empty_cache()

          all_scores = torch.stack(all_scores)

          # Affinity score to distance score
          pairwise_distances = 1 - all_scores.view(number_of_mentions, number_of_mentions).numpy()

          predicted = clustering.fit(pairwise_distances)
          predicted_clusters = predicted.labels_ + max_cluster_id
          max_cluster_id = max(predicted_clusters) + 1

          doc_ids.extend(doc_id[span_indices.cpu()])
          sentence_ids.extend(sentence_id[span_indices].tolist())
          starts.extend(start[span_indices].tolist())
          ends.extend(end[span_indices].tolist())
          all_topic_predicted_clusters.extend(predicted_clusters)
          torch.cuda.empty_cache()


    all_clusters = {}
    for i, cluster_id in enumerate(all_topic_predicted_clusters):
        if cluster_id not in all_clusters:
            all_clusters[cluster_id] = []
        all_clusters[cluster_id].append(i)


    if not config['use_gold_mentions']:
        all_clusters, doc_ids, starts, ends = remove_nested_mentions(all_clusters, doc_ids, starts, ends)

    all_clusters = {cluster_id:mentions for cluster_id, mentions in all_clusters.items() if len(mentions) > 1}

    print('Saving conll file...')
    '''
    doc_name = '{}_{}_{}_{}_model_{}_corpus_level.conll'.format(
        config['split'], config['mention_type'], config['linkage_type'], config['threshold'], config['model_num'])
    print(doc_name)
    pred_file = os.path.join(config['save_path'], doc_name)
    gold_file = '{}_{}_corpus_level.conll'.format(config['split'], config['mention_type'])
    gold_file = os.path.join(config['data_folder'], '../gold', gold_file)
    # write_output_file(data.documents, all_clusters, doc_ids, starts, ends, config['save_path'], doc_name,
    #                  topic_level=config.topic_level, corpus_level=not config.topic_level)
    
    # Split the corpus-level .conll file into separate document-level .conll files
    pred_out_dir = os.path.join(config['save_path'], 'pred_conll')
    gold_out_dir = os.path.join(config['data_folder'], 'gold_conll')
    if not os.path.isdir(pred_out_dir):
        os.mkdir(pred_out_dir)

    if not os.path.isdir(gold_out_dir):
        os.mkdir(gold_out_dir)
    prev_id = ''
    pred_dict = {}
    gold_dict = {}
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            doc_id = parts[2]
            
            if not doc_id == prev_id:
                prev_id = doc_id
                pred_dict[doc_id] = line
            else:
                pred_dict[doc_id] += line
    
    for doc_id in pred_dict:
        cur_pred = os.path.join(pred_out_dir, '{}.response'.format(doc_id))
        if not os.path.isfile(cur_pred):
            with open(cur_pred, 'w') as f:
                f.write(pred_dict[doc_id])
                
    prev_id = ''
    with open(gold_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            doc_id = parts[2]

            if not doc_id == prev_id:
                prev_id = doc_id
                gold_dict[doc_id] = line
            else:
                gold_dict[doc_id] += line

    for doc_id in gold_dict:
        cur_gold = os.path.join(gold_out_dir, '{}.key'.format(doc_id))
        if not os.path.isfile(cur_gold):
            with open(cur_gold, 'w') as f:
                f.write(gold_dict[doc_id])
    
    # Compute the document-level scores
    doc_ids = [k.split('.response')[0] for k in os.listdir(pred_out_dir)]
    metrics = ['bcub', 'ceafm', 'blanc']
    results = {m:[0., 0., 0.] for m in metrics}
    for doc_id in doc_ids:
        cur_pred = os.path.join(pred_out_dir, doc_id + '.response')
        cur_gold = os.path.join(gold_out_dir, doc_id + '.key')
        for metric in metrics:
            raw_out = subprocess.run(['perl', 'reference-coreference-scorers/scorer.pl', metric, cur_gold, cur_pred, 'none'], stdout=subprocess.PIPE)
            print(raw_out.stdout) # XXX
            coref_line = raw_out.stdout.strip().decode('utf-8').split('\n')[-2].split('\t')
            p = float(coref_line[0].split()[-1].split('%')[0])
            r = float(coref_line[1].split()[-1].split('%')[0])
            f1 = float(coref_line[2].split()[-1].split('%')[0])
            print('p, r, f1: {} {} {}'.format(p, r, f1)) # XXX
            results[metric][0] += p / len(doc_ids)
            results[metric][1] += r / len(doc_ids)
            results[metric][2] += f1 / len(doc_ids)
            # TODO Within-document clustering

    for metric in metrics:
        print('{} Precision: {}'.format(metric, results[metric][0]))
        print('{} Recall: {}'.format(metric, results[metric][1]))
        print('{} F1: {}'.format(metric, results[metric][2]))


    
