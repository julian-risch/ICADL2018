#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:22:57 2017

@author: Samuele Garda
"""
import logging
import argparse
import requests
import elasticsearch as es
from configparser import ConfigParser
import numpy as np
from load import ModelsLoader
from preprocess import preprocess_doc,FILTERS_FUNCS
from gensim.parsing import preprocessing as pp
from utils import string_to_dict


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO') 


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
  
  config = parse_arguments()
    
  models = ModelsLoader.load_models_from_config(config['LOADING'],config['PARAMETERS'])
  
  filter_funcs = FILTERS_FUNCS
    
  user_input = input("Make your query:\n")
  
  rm_sw = False if input("Stopwords were removed during training? [Yes/No]: \n").lower() == 'no' else True
  
  stem = False if input("Stemming was applied during training? [Yes/No]: \n").lower() == 'no' else True
  
  if rm_sw:
    filter_funcs.append(pp.remove_stopwords)
  if stem:
    filter_funcs.append(pp.stem_text)
  
  query = preprocess_doc(string_to_dict(user_input), filter_funcs = filter_funcs)['words']
    
  print(query)
  
  alphas_user = input("Try different starting learning rates (comma separated):\n")
  
  alphas = [float(a) for a in alphas_user.split(',')]
  
  steps_user = input("Try differnte number of steps for inference (comma separated): \n")
  
  steps = [int(s) for s in steps_user.split(',')]
  
  top_k = int(input("How many recommendations? : \n"))
  
  for model in models:
    
    for alpha in alphas:
      for step in steps:
        rec_vectors = model.infer_vector(doc_words = query,alpha = alpha, steps = step)
        reccomendations = model.docvecs.most_similar([rec_vectors], topn= top_k)
        print("{0} - alpha {1} - steps {2} :\n {3}".format(model,alpha,step,reccomendations))
        
        
      
  
  
  
  
  
  
  
  

    
    
    
    
    
    
  
  
