import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from itertools import combinations
import json
  
class TripletLoss(nn.Module):
  def __init__(self, config):
    super(TripletLoss, self).__init__()
    self.simtype = config.get('simtype', 'mean_max')

  def forward(self, text_outputs, image_outputs,        
              text_mask, image_mask,
              margin=1.):
    '''
    :param text_outputs: FloatTensor of size (batch size, max num. of tokens, text embed dim)
    :param image_outputs: FloatTensor of size (batch size, max num. of image regions, image embed dim)
    :return scores: FloatTensor of size (batch size, max num. of [score_type] spans, span embed dim)
    '''
    B, N, _ = text_outputs.size()
    L = image_outputs.size(1)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    ntokens = text_mask.sum(1).to(torch.int)
    nregions = image_mask.sum(1).to(torch.int)
    
    if B == 1:
      return loss
    for i in range(B):
      I_imp_ind = i
      S_imp_ind = i
      while I_imp_ind == i:
        I_imp_ind = np.random.randint(0, B)
      while S_imp_ind == i:
        S_imp_ind = np.random.randint(0, B)
      
      nT = ntokens[i]
      nTimp = ntokens[S_imp_ind]

      if len(nregions):
        nR = nregions[i]
        nRimp = nregions[I_imp_ind]

      anchorsim = self.matchmap_similarity(
          self.compute_matchmap(text_outputs[i, :nT], 
                                image_outputs[i, :nR])
      )
      Simpsim = self.matchmap_similarity(
          self.compute_matchmap(text_outputs[S_imp_ind, :nTimp],
                                image_outputs[i, :nR])
      )
      Iimpsim = self.matchmap_similarity(
          self.compute_matchmap(text_outputs[i, :nT],
                                image_outputs[I_imp_ind, :nRimp])
      )
      
      S2I_simdif = margin + Iimpsim - anchorsim
      if (S2I_simdif.data > 0).all():
        loss = loss + S2I_simdif
      I2S_simdif = margin + Simpsim - anchorsim
      if (I2S_simdif.data > 0).all():
        loss = loss + I2S_simdif
    loss = loss / B
    return loss
    
  def compute_matchmap(self, S, I):
    return torch.mm(S, I.t())

  def matchmap_similarity(self, M):
    if self.simtype == 'mean':
      return M.mean()
    elif self.simtype == 'mean_max':
      return M.max(1)[0].mean()
    elif self.simtype == 'max_mean':
      return M.max(0)[0].mean()

  def retrieve(self, text_outputs, image_outputs, text_masks, image_masks, k=10):
    n = len(text_outputs)
    nF = text_masks.sum(-1).to(torch.int)
    nR = image_masks.sum(-1).to(torch.int)

    S = torch.zeros((n, n), dtype=torch.float, requires_grad=False)
    for s_idx in range(n):
      for v_idx in range(n):
        S[s_idx, v_idx] = self.matchmap_similarity(
                            self.compute_matchmap(text_outputs[s_idx][:nF[s_idx]],
                                                  image_outputs[v_idx][:nR[v_idx]])
                            )

    _, I2S_idxs = S.topk(k, 0)
    _, S2I_idxs = S.topk(k, 1)
    return I2S_idxs.t(), S2I_idxs

class PredictionNetwork(nn.Module):
  def __init__(self,
               nPredicts,
               dimOutputAR,
               dimOutputEncoder,
               dropout=False,
               sizeInputSeq=116)
    super(PredictionNetwork, self).__init__()
    self.predictors = nn.ModuleList()
    self.RESIDUAL_STD = 0.01
    self.dimOutputAR = dimOutputAR
    
    self.dropout = nn.Dropout(p=0.5) if dropout else None
    for i in range(nPredicts):
      self.predictors.append(
        nn.Linear(dimOutputAR, dimOutputEncoder, bias=False)
      )
      if dimOutputEncoder > dimOutputAR:
        residual = dimOutputEncoder - dimOutputAR
        self.predictors[-1].weight.data.copy_(torch.cat([torch.randn(
            dimOutputAR, dimOutputAR), self.RESIDUAL_STD * torch.randn(residual, dimOutputAR)], dim=0))
        
  def forward(self, c, candidates):
    
    assert(len(candidates) == len(self.predictors))
    out = []
    
    for k in range(len(self.predictors)):
            locC = self.predictors[k](c)
            if self.dropout is not None:
                locC = self.dropout(locC)
            locC = locC.view(locC.size(0), 1, locC.size(1), locC.size(2))
            outK = (locC*candidates[k]).mean(dim=3)
            out.append(outK)
    return out 

class CPCLoss(nn.Module):
  def __init__(self,
               nPredicts,
               dimOutputAR,
               dimOutputEncoder,
               negativeSamplingExt,
               auxiliaryEmbedding=0,
               nAuxiliary=0
               startOffset=1):
    self.nPredicts = nPredicts
    self.startOffset = startOffset 
    if auxiliaryEmbedding > 0:
      print(
          f"Using {auxiliaryEmbedding}-dim auxiliary embeddings for {nAuxiliary} types")
      self.auxEmb = torch.nn.Embedding(nAuxiliary, auxiliaryEmbedding) # Auxiliary embedding to the context vector
    else:
      self.auxEmb = None

    self.wPrediction = PredictionNetwork(
        nPredicts, dimOutputAR, dimOutputEncoder,
        dropout=dropout, sizeInputSeq=sizeInputSeq - nPredicts)
    self.nPredicts = nPredicts
    self.negativeSamplingExt = negativeSamplingExt
    self.lossCriterion = nn.CrossEntropyLoss()

  def sampleClean(self, encodedData, windowSize):
    """
    :param encodedData: FloatTensor of size (batchSize, nNegativeExt, dimEncoded),
    :param windowSize: int, length of the positive sample sequence
    :param startOffset: int, offset of the positive samples relative to its frame
    :return outputs: list of nPredicts FloatTensors of size (batchSize, negativeSamplingExt+1, windowSize, dimEncoded), 
                     the concatenated features of both positive and negative samples
    """
    batchSize, nNegativeExt, dimEncoded = encodedData.size()
    outputs = []

    # (batchSize * nNegativeExt, dimEncoded)
    negExt = encodedData.contiguous().view(-1, dimEncoded)
    
    # Randomly draw (batchSize * negativeSamplingExt * windowSize) negative samples, so negativeSamplingExt negative samples per frame within the positive sample window by:
    # 1) Sample the batch index of the neg samples
    batchIdx = torch.randint(low=0, high=batchSize,
    ) 
     
    # 2) Sample the seq. index of the neg samples    
    seqIdx = torch.randint(low=1, high=nNegativeExt,
                           size=(self.negativeSamplingExt
                                 * windowSize * batchSize, ),
                           device=encodedData.device)
     
    # 3) Sample the index of the positive sample that the seq. index starts from (circular shift the global seqIdx if it exceeds the negative sample window) 
    baseIdx = torch.arange(0, windowSize, device=encodedData.device)
        baseIdx = baseIdx.view(1, 1,
                               windowSize).expand(1,
                                                  self.negativeSamplingExt,
                                                  windowSize).expand(batchSize, self.negativeSamplingExt, windowSize)

    seqIdx += baseIdx.contiguous().view(-1)
    seqIdx = torch.remainder(seqIdx, nNegativeExt)

    # 4) Compute the final index
    extIdx = seqIdx + batchIdx * nNegativeExt
    negExt = negExt[extIdx].view(batchSize, self.negativeSamplingExt,
                                 windowSize, dimEncoded)

    labelLoss = torch.zeros((batchSize * windowSize),
                            dtype=torch.long,
                            device=encodedData.device)
    
    for k in range(self.startOffset, self.nPredicts + self.startOffset):
        # Positive samples
        if k < self.nPredicts:
            posSeq = encodedData[:, k:-(self.nPredicts-k)]
        else:
            posSeq = encodedData[:, k:]

        posSeq = posSeq.view(batchSize, 1, posSeq.size(1), dimEncoded)
        fullSeq = torch.cat((posSeq, negExt), dim=1)
        outputs.append(fullSeq)

    return outputs, labelLoss
 
  def getInnerLoss(self):

    return "orthoLoss", self.orthoLoss * self.wPrediction.orthoCriterion()

  def forward(self, cFeature, encodedData, mask, label):
    """
    :param cFeature: FloatTensor of size (batchSize, seqSize, dimAR), contextualized features
    :param encodedData: FloatTensor of size (batchSize, seqSize, dimEncoded), input encoder features, 
    :param mask: FloatTensor of size (batchSize, negativeSamplingExt), mask on the input encoder features
    """
    batchSize, seqSize, dimAR = cFeature.size()
    windowSize = seqSize - self.nPredicts + (1 - self.startOffset)

    cFeature = cFeature[:, :windowSize]

    sampledData, labelLoss = self.sampleClean(encodedData, windowSize)
    
    if self.auxEmb is not None:
      l_ = label.view(batchSize, 1).expand(batchSize, windowSize)
      auxEmb = self.auxEmb(l_)
      cFeature = torch.cat([cFeature, auxEmb], dim=2)
    
    predictions = self.wPrediction(cFeature, sampledData)

    outLosses = [0 for x in range(self.nPredicts)]
    outAcc = [0 for x in range(self.nPredicts)]

    # TODO Handle mask
    for k, locPreds in enumerate(predictions[:self.nPredicts]):
        locPreds = locPreds.permute(0, 2, 1)
        locPreds = locPreds.contiguous().view(-1, locPreds.size(2))
        lossK = self.lossCriterion(locPreds, labelLoss)
        outLosses[k] += lossK.view(1, -1)
        _, predsIndex = locPreds.max(1)
        outAcc[k] += torch.sum(predsIndex == labelLoss).float().view(1, -1)

    return torch.cat(outLosses, dim=1), \
        torch.cat(outAcc, dim=1) / (windowSize * batchSize)
  
  def predict_clusters(cFeature, encodedData,
                       firstIdx, secondIdx):
      '''
      :param cFeature: FloatTensor of size (num. of spans, span embed dim),
      :param encodedData: FloatTensor of size (num. of spans, span embed dim),
      :param firstIdx: LongTensor of size (num. of mention pairs,)
      :param secondIdx: LongTensor of size (num. of mention pairs,)
      :return scores: FloatTensor of size (batch size, max num. of mention pairs),
      :return clusters: dict of list of int, mapping from cluster id to mention ids of its members 
      '''
      device = cFeature.device
      thres = 0
      span_num = max(secondIdx) + 1
      span_mask = torch.ones(len(firstIdx)).to(device)
      first_span_embeddings = cFeature[firstIdx]
      second_span_embeddings = encodedData[secondIdx]
      scores = self.cPredictor(first_span_embeddings, [second_span_embeddings]*self.nPredicts)[0]

      antecedents = -1 * np.ones(span_num, dtype=np.int64)

      # Antecedent prediction
      for idx2 in range(span_num):
        candidate_scores = []
        for idx1 in range(idx2):
          score_idx = idx1 * (2 * span_num - idx1 - 1) // 2 - 1
          score_idx += (idx2 - idx1)
          score = scores[score_idx].squeeze().cpu().detach().data.numpy()
          candidate_scores.append(score)

        if len(candidate_scores) > 0:
          candidate_scores = np.asarray(candidate_scores)
          max_score = candidate_scores.max()
          if max_score > thres:
            antecedent = np.argmax(candidate_scores)
            antecedents[idx2] = antecedent
      
      return antecedents, scores


      

