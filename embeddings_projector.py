#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:32:43 2017
@author: Samuele Garda
"""

import argparse
import os
import logging
import numpy as np
from load import ModelsLoader

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')  

def prune_name(name):
  
  return name.replace('/','_').replace('Doc2Vec(','').replace(')','').replace(',','_')


def parse_arguments():
  
  parser = argparse.ArgumentParser(description='Generate vector file and metadata file for doc2vec models to be used for visualization with \
                                   `http://projector.tensorflow.org/`')
  parser.add_argument('-d', '--dir', required=True, help='Path to model')
  parser.add_argument('-o', '--out-dir', default = './projector', help='Path where to store configuration file for tensorflow projector')
  parser.add_argument('-n', '--n-docs', default = None,type = int, help='Number of documents to visualize in projector. If set doc vectors will be randomly sampled \
                      from model to avoid sorted order.')
  parser.add_argument('-f', '--fields', required=True,type = str,nargs = '*', help='Names of tags to use as metadata. E.g. `index url`')
  
  args = parser.parse_args()
  
  return args


if __name__ == "__main__":
  
  args = parse_arguments()
  
  logging.getLogger("gensim").setLevel(logging.WARNING)

  models = ModelsLoader.load_from_dir(args.dir)
  
  logger.info("Creating main folder : {}\n".format(args.out_dir))
  if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
  
  for idx,model in enumerate(models):
    
    logger.info("Storing vectors and metadata for : {}".format(model))
    
    tot_vecs = len(model.docvecs) if not args.n_docs else args.n_docs
    
    model_name = prune_name(str(model))
    
    data_path = os.path.join(args.out_dir,model_name) 
    
    if not os.path.exists(data_path):
      os.mkdir(data_path)
    
    vector_data = os.path.join(data_path,'{}.vecs'.format(model_name))
    metadata = os.path.join(data_path,'{}.metadata'.format(model_name))
        
    with open(vector_data,'w+') as v_f, open(metadata,'w+') as m_f:
      m_f.write('{}\n'.format('\t'.join(args.fields)))
      for i in range(tot_vecs):
        if args.n_docs:
          i = np.random.randint(len(model.docvecs))
        doctag = '\t'.join(model.docvecs.index_to_doctag(i).split('-'))
        vector = '\t'.join(str(x) for x in model.docvecs.doctag_syn0[i])
        v_f.write('{}\n'.format(vector))
        m_f.write('{}\n'.format(doctag))
        
    logger.info("Saved {} vectors to `{}`".format(tot_vecs, vector_data))
    logger.info("Saved document metadata to `{}`\n".format(metadata))
        