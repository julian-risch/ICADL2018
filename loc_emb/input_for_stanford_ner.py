#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:11:20 2018

@author: Samuele Garda
"""

import argparse
import logging 
import spacy

SPACY_PARSER = spacy.load('en', disable = ["tagger","parser","ner"])

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO') 

def parse_args():
  parser = argparse.ArgumentParser(description='Create input file for Stanford NER')
  parser.add_argument("-i","--input",choices = ["books","movies"],required = True, help = "Type of dataset")
  parser.add_argument("-s","--source",required = True, help = "Path to dataset")
  parser.add_argument("-t","--target",type = str,required = True, help = "Output file")
  parser.add_argument("-l","--limit",type = int, default = 100, help = "# of review to parse")
  return parser.parse_args()

def write_books_summs(file, out_file ,limit = None):
  """
  Load book summaries from CMU Book Summaries Dataset.
  
  :params:
    file (str) : path to file
    limit (int) : # of summaries to load
  :return:
    df (pandas.Dataframe) : contains summaries and movie title    
  """
  
  logger.info("Loading book summaries from `{}`".format(file))
  
  limit = limit if limit else None
  
  with open(file) as infile, open(out_file, 'w') as out:
    for idx,line in enumerate(infile):
      wiki_id,freebase_id,title,auth,pub_data,genres,summ = line.split('\t')
      
      doc = SPACY_PARSER(summ.strip())
      summ = ' '.join([w.text for w in doc])
      out.write(summ +  '\n')
      if limit:
        if idx >= limit:
          break

if __name__ == "__main__":
  
  args = parse_args()
  
  if args.input == "books":
    
    write_books_summs(args.source,out_file = args.target, limit = args.limit)
    
  elif args.input == "movies":
  
    raise NotImplementedError
  
  
