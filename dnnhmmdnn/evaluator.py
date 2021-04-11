import torch
import collections
from allennlp_models.coref.metrics.conll_coref_scores import ConllCorefScores

class Evaluation:
    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels
        self.tp = (predictions == 1) * (labels == 1)

        self.tp_num = self.tp.sum().to(torch.float) # nonzero().squeeze().shape[0]
        self.tn = (predictions != 1) * (labels != 1)
        self.tn_num = self.tn.sum().to(torch.float) # nonzero().squeeze().shape[0]
        self.fp = (predictions == 1) * (labels != 1)
        self.fp_num = self.fp.sum().to(torch.float) # nonzero().squeeze().shape[0]
        self.fn = (predictions != 1) * (labels == 1)
        self.fn_num = self.fn.sum().to(torch.float) # nonzero().squeeze().shape[0]
        self.total = len(labels)



        self.precision = self.tp_num / (self.tp_num + self.fp_num) if self.tp_num + self.fp_num != 0 else torch.zeros(1, device=predictions.device)
        self.recall = self.tp_num / (self.tp_num + self.fn_num) if self.tp_num + self.fn_num != 0 else torch.zeros(1, device=predictions.device)


    def get_fp(self):
        return self.fp.nonzero().squeeze()

    def get_tp(self):
        return self.tp.nonzero().squeeze()

    def get_tn(self):
        return self.tn.nonzero().squeeze()

    def get_fn(self):
        return self.fn.nonzero().squeeze()

    def get_accuracy(self):
        return (self.tp_num + self.tn_num) / self.total

    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall

    def get_f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) > 0 else torch.zeros(1, device=self.precision.device)

class RetrievalEvaluation:
  def __init__(self, predictions, labels=None):
    self.predictions = predictions
    self.labels = labels
    self.total = len(predictions)
    if labels is None:
      self.labels = list(range(self.total))

  def get_recall_at_k(self, k=10):
    recall_at_k = 0.
    for i in range(self.total):
      for idx in range(k):
        if self.predictions[i][idx] == self.labels[i]:
          recall_at_k += 1
          break
    return recall_at_k / self.total  
         
class CoNLLEvaluation:
  def __init__(self):
    self.scorer = ConllCorefScores()

  def __call__(
    self,
    top_spans,
    predicted_antecedents,
    gold_spans,
    gold_labels
  ):
    """
    :param top_spans: LongTensor of size (num. spans, 2)
    :param predicted_antecedents: LongTensor of size (num. spans,)
    :param gold_labels: LongTensor of size (num. spans,)
    """
    num_spans = top_spans.size(0)
    antecedent_indices = torch.LongTensor([[j if j < i else -1 for j in range(num_spans)] for i in range(num_spans)])
    gold_clusters = self.get_gold_clusters(gold_spans, gold_labels)
    pred_clusters = self.get_predicted_clusters(top_spans, 
                                                antecedent_indices,
                                                predicted_antecedents)
    
    antecedent_indices = antecedent_indices.unsqueeze(0)
    top_spans = top_spans.unsqueeze(0).cpu()
    predicted_antecedents = predicted_antecedents.unsqueeze(0).cpu()
    
    metadata_list = [{'clusters': gold_clusters}]

    self.scorer(top_spans, antecedent_indices, predicted_antecedents, metadata_list)
    return pred_clusters, gold_clusters

  def get_predicted_clusters(self, top_spans, antecedent_indices, predicted_antecedents): 
    predicted_clusters, mention_to_pred = self.scorer.get_predicted_clusters(
                                          top_spans, 
                                          antecedent_indices, 
                                          predicted_antecedents)
    return predicted_clusters
     
  def get_gold_clusters(self, gold_spans, gold_labels):
    gold_spans = gold_spans.cpu().detach().numpy().tolist()  
    gold_labels = gold_labels.cpu().detach().numpy().tolist()
    cluster_dict = collections.defaultdict(list)
    
    for cluster_id, span in zip(gold_labels, gold_spans):
        if cluster_id > 0:
            cluster_dict[cluster_id].append(span)
    return list(cluster_dict.values())

  def get_metrics(self):
    muc = (self.scorer.scorers[0].get_precision(), self.scorer.scorers[0].get_recall(), self.scorer.scorers[0].get_f1())
    b_cubed = (self.scorer.scorers[1].get_precision(), self.scorer.scorers[1].get_recall(), self.scorer.scorers[1].get_f1())
    ceafe = (self.scorer.scorers[2].get_precision(), self.scorer.scorers[2].get_recall(), self.scorer.scorers[2].get_f1())
    avg = self.scorer.get_metric()
    return muc, b_cubed, ceafe, avg

  def make_output_readable(self, pred_clusters, gold_clusters, tokens, arguments=None):
    if arguments:
      pred_clusters_str = [[' '.join(tokens[m[0]:m[1]+1]) for m in cluster] for cluster in pred_clusters]
      gold_clusters_str = [[[' '.join(tokens[m[0]:m[1]+1]), [' '.join(tokens[a[0]:a[1]+1]) for a in arguments.get(tuple(m), [])]] for m in cluster] for cluster in gold_clusters]
    else:
      pred_clusters_str = [[' '.join(tokens[m[0]:m[1]+1]) for m in cluster] for cluster in pred_clusters]
      gold_clusters_str = [[' '.join(tokens[m[0]:m[1]+1]) for m in cluster] for cluster in gold_clusters]
    return pred_clusters_str, gold_clusters_str
