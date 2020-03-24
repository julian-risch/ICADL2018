#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:56:24 2018

@author: Samuele Garda
"""

import json
import os
import logging
import spacy
import argparse
import numpy as np
from collections import Counter
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO') 

def parse_args():
  parser = argparse.ArgumentParser(description='Create working environment for experiments with location in document embeddings')
  parser.add_argument("-i","--input",choices = ["books","movies"],required = True, help = "Type of dataset")
  parser.add_argument("-w", "--work-dir", required = True, type = str, help = "Folder where all stuff for experiments is stored")
  parser.add_argument("-f","--file",required = True, help = "Path to dataset: if used with `--create-input` must be raw data, else must be parsed file")
  parser.add_argument("-e","--emb", required = True,help = "Path to embeddings")
  parser.add_argument("-m", "--min-count" , default = 10, type = int, help = "Min count for location tokens")
  parser.add_argument("--create-input", default = None, help = "Name of folder where to store parsed file. Annotate with Name Entities input")
  parser.add_argument("--create-vocab", default = None, help = "Name of foldere where to create vocabulary and word index from parsed input")
  parser.add_argument("--create-matrices", default = None, help = "Name of folder where to create embedding matrices from input")
  
  
  return parser.parse_args()


def create_dir(path):
  """
  Create directory if does not exist.
  
  :params:
    path (str) : path of dir to create
  """
  if not os.path.exists(path):
    os.mkdir(path)
    
    
def join_path(paths):
  """
  Join list of paths
  
  :params:
    paths (list) : path hierarchy
    
  :return:
    path (str) : joined path
  """
  
  path = os.path.join(*paths)
  
  return path
    
def load_json_data(data_path):
  """
  Load JSON file (one item per line)
  
  :params:
    data_path (str) : path to JSON file
    
  :return:
    
    data (list) : list of dictionaries
  """
  
  data = []
  
  with open(data_path) as infile:
    for line in infile:
      file = json.loads(line)
      data.append(file)
      
  return data
    

def load_books(file,limit = None):
  """
  Load book summaries from CMU Book Summaries Dataset.
  
  :params:
    file (str) : path to file
    limit (int) : # of summaries to load
  :return:
    df (pandas.Dataframe) : contains summaries and movie title    
  """
  
  logger.info("Loading book summaries from `{}`".format(file))
  
  data = []
  
  limit = limit if limit else None
  
  with open(file) as infile:
    for idx,line in enumerate(infile):
      wiki_id,freebase_id,title,auth,pub_data,genres,summ = line.split('\t') 
      book = {}               
      book['title'] = title
      book['summ'] = summ
      book['genre'] = json.loads(genres) if genres else ''
      data.append(book)
      if limit:
        if idx >= limit:
          break
        
  logger.info("Loaded {} book summaries".format(len(data)))
  
  return data


def get_sets_from_sent(sent):
  """
  Split sentence in normal words and location words via spacy NER.
  
  :params:
    sent (str) : sentence
  :return:
    doc_locs,doc_other (tuple) : couple of list containing location tokens and normal words
  """
  
  doc = PARSER(sent)
  
  doc_locs = list(set([words for w in doc.ents if w.label_ in LOC for words in w.text.lower().split()]))
  
  doc_other = list(set([w.text.lower() for w in doc if not w.ent_type_ == LOC]))
    

  return doc_locs,doc_other

def sent_to_indices(sent, w2i):
  """
  Map list of tokens to their ids
  
  :params:
    sent (str) : sentence
    w2i (dict) : dictionary mapping tokens to id
    
  :return:
    list of ids
  """
  
  return [w2i.get(w,w2i['UNK']) for w in sent]


def build_vocab(texts, min_count):
  """
  Create vocabulary, word2index lookup table and inverse index2word from list of texts.
  Texts MUST be list of tokens.
  
  :param:
    texts (list) : list of lists of tokens
  
  :return:
    vocab (set) : vocabulary
    word2index (dict) : lookup token -> id
    index2word (dict) : lookup id -> token
  """
  
  vocab_counts = Counter([w for text in texts for w in text])
  
  vocab = set(k for k,v in vocab_counts.items() if v > min_count) if min_count else set(vocab_counts.keys())
  
  word2idx = {w : idx for idx,w in enumerate(vocab,start = 1)}
  
  word2idx['UNK'] = 0
  
  idx2word = {idx : w for w,idx in word2idx.items()}
  
  return vocab,word2idx,idx2word


def load_emb_lookup(we_file,limit = None):
  """
  Load pre-trained embeddings from file (IN WORD2VEC FORMAT).
  
  :params:
    we_file (str) : path to word embeddings file
    limit (int) : # of embeddings to load
  
  :return:
    embeddings_dict (dict) : lookup table word -> embedding
    
  """
  
  logger.info("Loading pre-trained word embeddings from `{}`".format(we_file))
  
  embeddings_dict = {}
  
  limit = limit if limit else None

  with open(we_file) as we:
    for idx,line in enumerate(we):
      embeddings_dict[line.split()[0]] = np.asarray(line.split()[1:], dtype = 'float32')
      if limit:
        if idx >= limit:
          break
  
  logger.info("Loaded {} pre-trained word embeddings".format(len(embeddings_dict)))
  return embeddings_dict
  
def get_embedding_matrix(emb_lookup,word2index,vocab):
  """
  Create embedding matrix from vocabulary. Words out of vocabulary are initialized with 0 vectors
  
  :param:
    emb_lookup (dict) : lookup table word -> embedding
    word2index (dict) : lookup token -> id
    vocab (set) : vocabulary
    emb_dim (int) : size of embeddings
    
  :return:
    embedding_matrix (numpy.ndarray) : embedding weights
  """
  
  found = 0
  
  emb_dim = len(next(iter(emb_lookup.values())))
  
  embedding_matrix = np.zeros([len(vocab)+1, emb_dim])
  for word, i in word2index.items():
    embedding_vector = emb_lookup.get(word)
    if embedding_vector is not None:
      found += 1
      embedding_matrix[i] = embedding_vector
    else:
      pass
    
  logger.info("{} pretrained word embeddings found".format(found))
    
  return embedding_matrix


if __name__ == "__main__":
  
  args = parse_args()
  
  PARSER = spacy.load('en', disable = ["tagger","parser"])
  LOC = ["DATE"] 
    
  logger.info("Creating working directory `{}`".format(args.work_dir))
  
  create_dir(args.work_dir)
  
  if args.input == "books":
    
    if args.create_input:
    
      data = load_books(args.file, limit = None)
    
    else:
      
      data_complete = load_json_data(args.file)
      
      docs_others = [doc['nonloc_tok'] for doc in data_complete]
      
      docs_locs = [doc['loc_tok'] for doc in data_complete]
    
  elif args.input == "movies":
  
    raise NotImplementedError
    
  emb_lookup = load_emb_lookup(args.emb)
  
  if args.create_input:
    
    data_path = join_path([args.work_dir,args.create_input]) 
    
    create_dir(data_path)
  
    docs_others = []
    
    docs_locs = []
    
    logger.info("Start processing documents... This might take a while")
    
    with open(os.path.join(data_path,'booksummaries_set.json'), 'w') as out_file:
      for idx,doc in enumerate(data, start = 1):
        
        doc_loc,doc_other = get_sets_from_sent(doc['summ'])
        doc['loc_tok'] = doc_loc
        doc['nonloc_tok'] = doc_other
        doc.pop('summ')
          
        book_data = json.dumps(doc) + '\n'
        
        out_file.write(book_data)
              
        docs_others.append(doc_other)
      
        docs_locs.append(doc_loc)
        
        if (idx%1000) == 0:
          logger.info("Parsed {} documents".format(idx))
  

  if args.create_vocab:
    
    vocabs_path = join_path([args.work_dir, args.create_vocab])
    
    create_dir(vocabs_path)
  
    logger.info("Created vocabulary for non-loc embeddings")
    
    other_vocab,other_w2i,other_i2w = build_vocab(docs_others, min_count = None)
    
    pickle.dump(other_vocab, open(os.path.join(vocabs_path,'non-loc.vocab'), 'wb'))
    
    pickle.dump(other_w2i, open(os.path.join(vocabs_path,'non-loc.w2i'), 'wb'))
    
    logger.info("Created vocabulary for loc embeddings")
    
    locs_vocab,locs_w2i,locs_i2w = build_vocab(docs_locs, min_count = args.min_count)
    
    pickle.dump(locs_vocab, open(os.path.join(vocabs_path,'loc.vocab'), 'wb'))
    
    pickle.dump(locs_w2i, open(os.path.join(vocabs_path,'loc.w2i'),'wb'))
    
    logger.info("Created vocabulary for loc embeddings")
    
  if args.create_matrices:
    
    numpys_path = join_path([args.work_dir, args.create_matrices])
    
    create_dir(numpys_path)
        
    others_emb_matrix = get_embedding_matrix(emb_lookup = emb_lookup,word2index = other_w2i, vocab = other_vocab)
    
    np.save(join_path([numpys_path, 'non-loc']), others_emb_matrix)
    
    locs_emb_matrix = get_embedding_matrix(emb_lookup = emb_lookup,word2index = locs_w2i, vocab = locs_vocab)
    
    np.save(join_path([numpys_path, 'loc']), locs_emb_matrix)
    
    logger.info("Created embedding matrices")
  