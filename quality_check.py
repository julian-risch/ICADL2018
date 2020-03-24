#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:54:54 2017

@author: Samuele Garda
"""
import logging
import numpy as np
from collections import OrderedDict,Counter
from ml_metrics import apk
from utils import majority_vote

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO') 


class QualityChecker(object):
  """
  Object performing quality check of document embeddings
  """
  
  def __init__(self,models,corpus):
    """
    Construct new object
    """
    self.corpus = corpus
    self.models = models if isinstance(models,dict) else OrderedDict((str(model).replace('/','-'),model) for model in models)
    self.rand_id = np.random.randint(len(corpus))
    
    
  def current_losses(self):
    """
    Log current loss for model.
    
    :param model: gensim.models
    
    """
    for name,model in self.models:
      logger.info("{0} current loss {1}".format(name,model.get_latest_training_loss()))
    
  def trained_most_similar(self,topk):
    """
    Log `n_top` most similar documents to given random document.
    
    :param model: gensim model
    :param corpus: corpus
    :param n_top: n most similar
    """
    for name,model in self.models.items():
      tag = self.corpus[self.rand_id].tags
      logger.info("{0} - training - most similar for {1} : {2}\n".format(name, tag, model.docvecs.most_similar(tag, topn=topk)))
    
  
  def inferred_most_similar(self,topk):
    """
    Log `n_top` most similar documents to inferred vector for words present in random document.
    
    :param models: gensim model
    :param corpus: corpus
    :param n_top: n most similar
    """
    for name,model in self.models.items():
      inferred_vector = model.infer_vector(self.corpus[self.rand_id].words)
      tag = self.corpus[self.rand_id].tags
      logger.info("{0} - inferred - most similar for {1} : {2}\n".format(name, tag, model.docvecs.most_similar([inferred_vector], topn=topk)))
      
      
  def base_check_from_config(self,config_train):
    
    log_train =  config_train.getint('quality_check_infered')
    log_inferred =  config_train.getint('quality_check_infered')
    
    if log_train:
      
      self.trained_most_similar(topk = log_train)
      
    if log_inferred:
      
      self.inferred_most_similar(topk = log_inferred)
      
  def count_ranks(self,models, corpus):
  
    for model_name,model in models.items():
      ranks = []
      for doc_id in range(len(corpus)):
        inferred_vector = model.infer_vector(corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        
      logger.info("Ranks count for {} : {}".format(model_name,dict(Counter(ranks))))

def mapk_train_vectors(models,labels, k=10):
  """
  Log model MAP@k scores of models.
  
  :params:
    
    models (dict) : dict (name,model)
    labels (dict) : dict mapping item to its recommendations
    k (int) : K in MAP@K
  
  """
  
  logger.info("Starting evaluation process for MAP@{}...".format(k))
  
  best_score =  0
  best_model = None
  
  for model in models:
  
    predictions = {}
    
    for doc_id in labels.keys():
      try:
        predictions[doc_id] = [l[0] for l in model.docvecs.most_similar(doc_id,topn = k)]
      except TypeError:
        pass
        
    mapk = np.mean([apk(labels[doc_id],predictions[doc_id],k) for doc_id in labels.keys() if doc_id in predictions.keys()])
    logger.info("{0} - MAP@{1} : {2}\n".format(str(model),k, mapk))
    
    if mapk > best_score:
      best_score = mapk
      best_model = str(model)
      
  logger.info("Best model with MAP@{0} = {1} : {2} \n ".format(k,best_score,best_model))
    
    
def evaluate_tpr_indices(models,corpus,topk):
  """
  Evaluate document embeddings with TPR. For each document in `test_set` infer vector and find `k` nearest document (cosine similarity).
  Take majority vote on the inferred labels.
  
  :params:
    
    models (dict) : dict (name,model) of gensim.Doc2Vec models to be tested
    test_set (list) : list of docuemnts in namedtuple format. Thet MUST have at least `words` and `tags` attributes
    topk (int) : how many documents to predict
  
  """
  
  logger.info("Starting evaluation process - TPR@{}\n".format(topk))
  
  tl_list = [doc.tags[0].split('\t')[0] for doc in corpus]
  
  unique_names = set(tl_list)
  
  label2index = {name : i for i,name in enumerate(unique_names,start = 1)}
  
  index2lable = {v : k for k,v in label2index.items()}
  
  true_labels = np.asarray([label2index[l] for l in tl_list])
  
  labels_count = Counter(true_labels)
  
  for model in models:
    
    logger.info("\nEvaluating `{}`".format(str(model)))
    
    logger.info("Inferring vectors for {} documents".format(len(corpus)))
    
    inferred_vectors = [model.infer_vector(doc.words) for doc in corpus]
    
    most_similars = [model.docvecs.most_similar([inf_vec], topn=topk) for inf_vec in inferred_vectors]
    
    all_votes = [[news.split('\t')[0] for (news,score) in ms] for ms in most_similars]
    
    predicted = np.asarray([label2index[majority_vote(votes)[0]] for votes in all_votes])
    
    scores = {}
    
    for label,count in labels_count.items():
      
      by_class_pred = np.where(predicted != label,0,label)
      
      hits = np.sum(by_class_pred == true_labels)
      
      scores[index2lable[label]] = hits / count
      
    logger.info("TPR@10 : {}".format(scores))
      
    logger.info("Average TPR@10 : {}".format(np.sum(predicted==true_labels)/len(true_labels)))
        
    

  
      
    
    
    
    
#    
#def get_most_median_least_similar(model,corpus,rand_id):
#  """
#  Print most,median,least similar documents to a random document.
#  
#  :param model: gensim model
#  :param corpus: corpus
#  """
#  sims = model.docvecs.most_similar(rand_id, topn=model.docvecs.count)
#  logger.info(u'TARGET (%d): «%s»\n' % (rand_id, ' '.join(corpus[rand_id].words)))
#  logger.info(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
#  for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#    logger.info(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(corpus[sims[index][0]].words)))
    
  
  
