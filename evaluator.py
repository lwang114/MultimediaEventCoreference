import torch


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



        self.precision = self.tp_num / (self.tp_num + self.fp_num) if self.tp_num + self.fp_num != 0 else 0
        self.recall = self.tp_num / (self.tp_num + self.fn_num) if self.tp_num + self.fn_num != 0 else 0


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
        return 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0

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
         
          


