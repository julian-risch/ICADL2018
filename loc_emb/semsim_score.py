#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:00:19 2018

@author: Samuele Garda
"""


import numpy as np
from nltk.corpus import wordnet as wn
from nltk import pos_tag

def penn_to_wn(tag):
  """ 
  Convert between a Penn Treebank tag to a simplified Wordnet tag 
  
  :params:
    tag (str) : Penn Treebank PoS tag
  
  :return:
    simplified PoS tag for WordNet
    
  """
  if tag.startswith('N'):
      return 'n'
 
  if tag.startswith('V'):
      return 'v'
 
  if tag.startswith('J'):
      return 'a'
 
  if tag.startswith('R'):
      return 'r'
 
  return None


def tagged_to_synset(word, tag):
  """
  Return first synsets for given word and tag
  
  :params:
    word (str) : word
    tag (str) : PoS tag
  
  :return:
    first synsets if found else None
  """
  wn_tag = penn_to_wn(tag)
  if wn_tag is None:
      return None
 
  try:
      return wn.synsets(word, wn_tag)[0]
  except IndexError:
      return None
  

def idf_score(w,D):
  """
  Compute IDF of word.
  :math:`ìdf(w,D)=\log(1+\\frac{N}{|d \in D : w \in d|})`
  
  :param:
    w (str) : word
    D (list) : list of documents (list of tokens)
  :return:
    idf (int) : IDF score
  """
  
  idf = np.log(len(D) / (1 + sum([1 for d in D if w in d])))
  
  return idf

def sentence_similarity(sentence1, sentence2):
  """ 
  Compute the sentence similarity using Wordnet.
  
  :params:
    sentence1 (list) : list of tokens
    sentence2 (list) : list of tokens
  
  :return:
    score (int) : similarity score 
  """
  # Tokenize and tag
  sentence1 = pos_tag(sentence1)
  sentence2 = pos_tag(sentence2)
 
  # Get the synsets for the tagged words
  synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
  synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
  
  synsets1 = [ss for ss in synsets1 if ss]
  synsets2 = [ss for ss in synsets2 if ss]
  
  score, count = 0.0, 0
 
  # For each word in the first sentence
  for synset in synsets1:
      # Get the similarity value of the most similar word in the other sentence
      scores = [synset.path_similarity(ss) for ss in synsets2]
      scores = [score for score in scores if score]
      if scores:
        score += max(scores)
        count += 1
 
  # Average the values
  score /= count
  return score


def sym_sent_sim(sentence1, sentence2):
    """ compute the symmetric sentence similarity using Wordnet """
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2 
  
  
if __name__ == "__main__":
  
  sent1 = "Dogs are awesome .".split()
  sent2 = "Cats are beautiful animals .".split()
  
  print("SentSim {}".format(sentence_similarity(sent1,sent2)))
  print("SymSentSim {}".format(sym_sent_sim(sent1,sent2)))