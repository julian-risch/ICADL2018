#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:49:01 2017

@author: Samuele Garda

Module containing preprocessing utilities.
"""

import logging
from collections import namedtuple
from gensim.parsing import preprocessing as pp

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO') 

FILTERS_FUNCS = [pp.strip_tags, pp.strip_punctuation, pp.strip_multiple_whitespaces]


def dict2namedtuple(dictionary):
  """
  Transoform dictionary into namedtuple.
  """
  return namedtuple('GenericDict', dictionary.keys())(**dictionary)

class DocumentsTagged(object):
  """
  Iterator of tagged documents to be fed to gensim Doc2Vec.
  """
  
  def __init__(self, documents):
    """
    Generate iterator of tagged documents.
    :param documents: tokenized documents to be labeled
    :type documents: generator of (list of lists)
    """
    self.documents = documents
    
    
  def __iter__(self):
    """
    Iterator of documents. Assign unique tag number to each document.
    :rtype: gensim.models.doc2vec.LabeledSentence
    """
    for doc in self.documents:
      yield namedtuple('TaggedDocument',doc.keys())(**doc)
      



def preprocess_doc(doc,filter_funcs):
  """
  Preprocess single document
  
  :params:
    doc (dict) : document
    filter_funcs (list) : list of gensim.parsing.preprocessing functions
  
  :return:
    document
  """
  
  doc['words'] = pp.preprocess_string(s = doc['words'], filters = filter_funcs)
  
  return doc 



def preprocess_data(data, rm_stopwords, stemming):
  """
  Preprocess docs in iterable
  
  :params:
    data (iterable) : documents
    rm_stopwords (bool) : remove stopwords
    stemming (bool) : apply stemming
    
  :return:
    generator
  """
  
  filter_funcs = FILTERS_FUNCS
  
  if rm_stopwords:
      
    filter_funcs.append(pp.remove_stopwords)
      
  if stemming:
      
    filter_funcs.append(pp.stem_text)
    
  for doc in data:
     
    doc = preprocess_doc(doc = doc, filter_funcs = filter_funcs)
    
    yield doc
     
  
