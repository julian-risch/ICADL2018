#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:41:53 2017

@author: Samuele Garda

Module containing class for training gensim.Doc2Vec models.
"""

import os
import numpy as np
import logging
import argparse
import requests
import elasticsearch as es
from configparser import ConfigParser
from collections import OrderedDict
from random import shuffle
from load import load_data_from_config,ModelsLoader,load_and_process
from preprocess import preprocess_data,DocumentsTagged
from quality_check import QualityChecker
from utils import timeit,elapsed_timer,downsample,Generator


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO') 

class ModelsTrainer:
  """
  Train gensim.Doc2Vec models. It is possible to specifify whether to manually controll the learning rate decay and shuffling documents for obtaining 
  better results. During training phase for each epoch for quality check a random document id is selected and is computed the most similar documents and 
  the most similar document for the vector inferred with the words present in the random document.
  """
  
  @classmethod  
  def _create_checkpoint(cls,checkpoint):
    """
    Create folders where to store chekpoints if not exisisting yet.
    """
    if not os.path.exists(checkpoint):
      os.makedirs(checkpoint)
      
  @classmethod   
  @timeit
  def init_models(cls,models,corpus):
    
    logger.info("Initializing models with corpus...")
    
    # create input layer
    models[0].build_vocab(corpus)
    # reset from firs model if more than 1
    if len(models) > 1:
      for model in models[1:]:
        model.reset_from(models[0])
        
    train_models = OrderedDict((str(model).replace('/','-'),model) for model in models)
    
    return train_models
  
  @staticmethod
  @timeit
  def train_manual_lr(models,corpus,epochs,shuffle_docs,alpha,minalpha,alpha_delta,checkpoint = None):
    """
    Train models given a corpus.
    
    :param models: models to be trained
    :type models: list
    :param corpus: documents for training
    :type corpus: list of TaggedDocuments
    
    :rtype: list
    """
    
    logger.info("Training models with maunally controlled learning rate decay...\n")
    
    original_alpha = alpha
    
    for model_name,model in models.items():
      
      logger.info("Training model {}".format(model_name))
      
      with elapsed_timer() as elapsed:
        
        for epoch in range(1,epochs+1):
          
          # fix alpha 
          model.alpha,model.minalpha = alpha,alpha
          
          if shuffle_docs:
              shuffle(corpus)
              
          model.train(corpus,total_examples=model.corpus_count,epochs = 1)
          
          # decay
          alpha -= alpha_delta
          
        duration = elapsed()
          
        logger.info("{0} completed training in {1:.3f}s\n".format(model_name,duration))
      
      # reset to original alpha for next model to train
      alpha = original_alpha
   
    if checkpoint:
      
      if not os.path.exists(checkpoint):
        logger.warning("Path provided for storing model is not valid : {}".format(checkpoint))
      
      for model_name,model in models.items():
        model.save("{}".format(os.path.join(checkpoint,model_name)))
        logger.debug("Saved chekpoint for {0} to {1}".format(model_name,checkpoint))
    else:
      logger.warning("No checkpoint specified! Model won't be saved at the end of training")       
  
  @staticmethod  
  @timeit    
  def train_standard(models,corpus,epochs,shuffle_docs,checkpoint = None):
  
    
    if shuffle_docs:
      shuffle(corpus)
      
    logger.info("Training models with standard learning rate decay...\n")
      
    for model_name,model in models.items():
      
      logger.info("Training model {}".format(model_name))
      with elapsed_timer() as elapsed:
        model.train(corpus,total_examples=model.corpus_count, epochs = epochs)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        duration = elapsed()
      logger.info("{0} completed training in {1:.3f}s\n".format(model_name,duration))
      
      
    if checkpoint:
      
      if not os.path.exists(checkpoint):
        logger.warning("Path provided for storing model is not valid : {}".format(checkpoint))
      
      for model_name,model in models.items():
        model.save("{}".format(os.path.join(checkpoint,model_name)))
        logger.debug("Saved chekpoint for {0} to {1}".format(model_name,checkpoint))
    else:
      logger.warning("No checkpoint specified! Model won't be saved at the end of training")
  
  @classmethod
  def train_from_config(cls,models,corpus,training_config):
    
    epochs = training_config.getint('epochs',10)
    checkpoint = training_config.get('checkpoint',os.path.join(os.getcwd(),'models'))
    shuffle_docs = training_config.getboolean('shuffle',True)
    
    cls._create_checkpoint(checkpoint)
    
    if training_config.getboolean('adapt_alpha'):
      
      alpha = training_config.getfloat('alpha',0.025)
      min_alpha = training_config.getfloat('min_alpha',0.001)
      
      # CHANGE HERE FOR DIFFERENT LEARNING RATE UPDATES
      alpha_delta = (alpha - min_alpha) / epochs
      
      
      cls.train_manual_lr(models = models, corpus = corpus, epochs = epochs, 
                           shuffle_docs = shuffle_docs, alpha = alpha, minalpha = min_alpha,
                           alpha_delta = alpha_delta, checkpoint = checkpoint)
          
    else:
        
        cls.train_standard(models = models, corpus = corpus, epochs = epochs,
                   shuffle_docs = shuffle,checkpoint = checkpoint)
        
        
  
def parse_arguments():
  """
  Read configuration file.
  """
  
  parser = argparse.ArgumentParser(description='Similarity queries in corpus with document vectors estimated via paragraph2vec model')
  parser.add_argument('-c', '--conf', required=True, help='Configuration file')
  args = parser.parse_args()
  
  configuration = ConfigParser(allow_no_value=False)
  configuration.read(args.conf)
  return configuration

if __name__ == "__main__":
  
  # ignore warning till have proper account on es
  es.connection.http_urllib3.warnings.filterwarnings('ignore')
  requests.packages.urllib3.disable_warnings()
  
  logging.getLogger("gensim").setLevel(logging.WARNING)
  logging.getLogger('elasticsearch').setLevel(logging.WARNING)
  
  np.random.seed(7)
  
  logger.info("\nStaring training process\n")
  
  config = parse_arguments()
  
  dataset = load_data_from_config(config['LOADING'])
    
  data_preprocessed = preprocess_data(dataset, 
                                      rm_stopwords=config['PREPROCESS'].getboolean('rm_stopwords',True),
                                      stemming = config['PREPROCESS'].getboolean('stem',True))
  
  data_doc2vec = DocumentsTagged(data_preprocessed)
  
  gen_data_doc2vec = Generator(data_doc2vec)
  
  corpus = load_and_process(gen_data_doc2vec) 
  
  
  if config['TRAIN'].getboolean('downsample'):
    corpus = downsample(corpus)
  
  models_to_train = ModelsLoader.load_models_from_config(config['LOADING'],config['PARAMETERS'])
  
  models = ModelsTrainer.init_models(models_to_train,corpus)

  ModelsTrainer.train_from_config(models,corpus,config['TRAIN'])
  
  q_checker = QualityChecker(models,corpus)
   
  q_checker.base_check_from_config(config['TRAIN'])
  