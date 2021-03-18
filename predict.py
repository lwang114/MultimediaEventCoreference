from sklearn.cluster import AgglomerativeClustering
import argparse
import pyhocon
from transformers import AutoTokenizer, AutoModel
from itertools import product
import collections
from tqdm import tqdm

from conll import write_output_file
from models import SpanScorer, SimplePairWiseClassifier, SpanEmbedder
from coref.utils import *
from coref.model_utils import *
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
    create_folder(config['model_path'])

    pred_out_dir = os.path.join(config['model_path'], 'pred_conll')
    gold_out_dir = os.path.join(config['data_folder'], '../gold_{}'.format(config['label_type']))
    if not os.path.isdir(pred_out_dir):
        os.mkdir(pred_out_dir)

    if not os.path.isdir(gold_out_dir):
        os.mkdir(gold_out_dir)

    # Compute the document-level scores
    doc_ids = ['_'.join(k.split('_')[:-2]) for k in os.listdir(pred_out_dir)]
    metrics = ['muc', 'bcub', 'ceafe', 'ceafm', 'blanc']
    results = {m:[0., 0., 0.] for m in metrics}
    n = len(doc_ids)
    for doc_id in doc_ids:
        print(doc_id)
        cur_pred = os.path.join(pred_out_dir, doc_id + '_corpus_level.conll')
        cur_gold = os.path.join(gold_out_dir, doc_id + '_corpus_level.conll')
        for metric in metrics:
            raw_out = subprocess.run(['perl', 'coref/reference-coreference-scorers/scorer.pl', metric, cur_gold, cur_pred, 'none'], stdout=subprocess.PIPE)
            coref_line = raw_out.stdout.strip().decode('utf-8').split('\n')[-2].split('\t')
            n_mentions = int(coref_line[0].split()[-2].split()[-1].split(')')[0])
            if n_mentions == 0:
              n -= 1
              break
            r = float(coref_line[0].split()[-1].split('%')[0])
            p = float(coref_line[1].split()[-1].split('%')[0])
            f1 = float(coref_line[2].split()[-1].split('%')[0])
            print('p, r, f1: {} {} {}'.format(p, r, f1)) # XXX
            results[metric][0] += p 
            results[metric][1] += r 
            results[metric][2] += f1

    for metric in metrics:
      for i in range(3):
        print(n)
        results[metric][i] /= max(n, 1)

    for metric in metrics:
        print('{} Precision: {}'.format(metric, results[metric][0]))
        print('{} Recall: {}'.format(metric, results[metric][1]))
        print('{} F1: {}'.format(metric, results[metric][2]))


    
