#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:46:31 2017

@author: Samuele Garda
"""

import logging
import time
from contextlib import contextmanager
from timeit import default_timer
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from collections import Counter

#def get_k_tfidf_vec(data,k): 
#  docuemnts = [doc['text'] for doc in data ]
#  vectorizer = TfidfVectorizer(analyzer = 'word', preprocessor = ' '.join, tokenizer= str.split )
#  tfidf_matrix = vectorizer.fit_transform(docuemnts).toarray()
#  k_tfidf_matrix = np.partition(tfidf_matrix,-k)[:,-k:]
#  print(k_tfidf_matrix)

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO') 


class Generator:
  """
  Wrapper class for getting return from generator
  """
  def __init__(self, gen):
      self.gen = gen

  def __iter__(self):
      self.value = yield from self.gen

def timeit(method):
  """
  Decorator for timing functions call
  """
  def timed(*args, **kw):
      ts = time.time()
      result = method(*args, **kw)
      te = time.time()

      if 'log_time' in kw:
          name = kw.get('log_name', method.__name__.upper())
          kw['log_time'][name] = int((te - ts) * 1000)
      else:
          logger.info('%r  %2.2f s' %(method.__name__, (te - ts)))
      return result

  return timed
  
  
@contextmanager
def elapsed_timer():
  """
  Helper function for timing.
  """
  start = default_timer()
  elapser = lambda: default_timer() - start
  yield lambda: elapser()
  end = default_timer()
  elapser = lambda: end-start
  
  
def string_to_dict(string):
  """
  Convert string to dict with `ẁords` as key. Compatibility with preprocessing pipeline
  
  :param string: string of words
  :type string: str
  :rtype: dict
  """
  
  return {'words' : string}


def majority_vote(seq):
  """
  Find max occurring value in a list
  """
  c = dict()
  for item in seq:
      c[item] = c.get(item, 0) + 1
  return max(c.items(), key=itemgetter(1))

  
def downsample(corpus):
  """
  Downsample training corpus to category with minium count
  
  :params:
    corpus (list) : list of namedtuple with `tag` attribute
  :return:
    resampled_data (list) : list of namedtuple. Resampled corpus with # docs = min count tag for each tag
  """
  
  resampled_data = []
  
  all_tags = [doc.tags[0].split('\t')[0] for doc in corpus ]
  
  corpus_counts = dict(Counter(all_tags))
  
  min_tag = min(corpus_counts, key=corpus_counts.get)

  for tag in corpus_counts:
    
    by_tag_split = [doc for doc in corpus if doc.tags[0].split('\t')[0] == tag]
    if tag != min_tag:
      resampled = resample(by_tag_split, n_samples = corpus_counts[min_tag], replace = False ,random_state = 42)
      resampled_data += resampled
    else:
      resampled_data += by_tag_split
      
  logger.info("Downsampled training corpus to less frequent category `{0}`( {1} ) - New size : {2}".format(min_tag,
              corpus_counts[min_tag],len(resampled_data)))   
  
  return resampled_data
      
        

  
    
    
      
        
      
      
      
      
      
  