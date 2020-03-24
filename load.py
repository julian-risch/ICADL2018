#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:36:46 2017

@author: Samuele Garda

Module containing loading classes for:
  - ES/JSON documents
  - gensim models
"""

import sys
import time
import logging
import glob
import json
import os
from elasticsearch import Elasticsearch,helpers
from gensim.models import doc2vec
from sklearn.model_selection import ParameterGrid
from multiprocessing import cpu_count
from utils import timeit
from collections import defaultdict
#from xml.etree import ElementTree as ET


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO') 

def _show_loading_buffer(index):
  """
  Prints a dot for every 1000 documents loaded in stdout.
  
  :param index: document index
  :type index: integer
  """
  if index % 1000 == 0:
      sys.stdout.write('.')
      sys.stdout.flush()
      time.sleep(.1)
 

def _new_logging_block():
  """
  Prints a `\n` in stdout.
  """
  sys.stdout.write('\n')
  sys.stdout.flush()
    
    

def load_es_by_index(es_server,es_indices,es_query,doc_ids,doc_texts):
  """
  Generator of documents from ES index. Fields of interest must be provided in form of list.
  ORDER MATTERS! I.e. first `doc_ids` and `doc_texts` must correspond to first `es_indices`
  
  :params:
    es_server (list) : server where es instance is running
    es_indices (list) : index where to extract data
    es_query (dict) : es formatted query
    doc_ids (list) : fileds to be used as id for document
    doc_text (list) : fileds where text to be processed is stored
      
  :return: generator
  """
  
  doc_counts = {}
  
  ES = Elasticsearch(es_server, verify_certs=False,use_ssl = True, http_auth= "admin:admin")
        
  for idx,es_index in enumerate(es_indices):
    
    logger.info("Loading documents from `{0}/{1}`".format(es_server,es_index))
    
    missing_field =  0
    
    doc_counts[es_index] = 0
    
    results_gen = helpers.scan(ES, query = json.loads(es_query), index=es_index, scroll = '30s')
    
    for i,doc in enumerate(results_gen):
      
      try:
        
        # keep source
        doc = doc['_source']
        
        # change dict keys for compatibility with gensim data structure (TaggedDocument)
        doc['tags'] = ["{}\t{}".format(es_index,doc.pop(doc_ids[idx]))]
        doc['words'] = doc.pop(doc_texts[idx])
        
        if doc['words'] != None:
          
          doc_counts[es_index] += 1
          
          yield doc
          
        else:
          
          missing_field += 1
      
      except KeyError:
        
        missing_field += 1
 
      _show_loading_buffer(i)
    
    _new_logging_block()
    
    logger.info("Documents in `{}` : {} - missing field : {}".format(es_index,doc_counts[es_index],missing_field))
  
  _new_logging_block()
  
    
def load_es_by_class(es_server,es_index,es_query,doc_ids,doc_texts,classes):
  """
  Load documents from ES instance by class. MUST BE SINGLE INDEX. For each specified class query is performed to index
  to retireve documents. Default field and query in `query string` are MODIFIED internally.
  DO NOT MODIFY QUERY SKELETON (FIRST PART OF MUST).
  
  :params:
    es_server (list) : server where es instance is running
    es_index (list) : index where to extract data
    es_query (dict) : es formatted query
    doc_ids (list) : fileds to be used as id for document
    doc_text (list) : fileds where text to be processed is stored
    classes (list) : list of classes (regexp for query)
      
  :return: generator
  
  """
  
  logger.info("Loading documents from `{0}/{1}`".format(es_server,es_index))
  
  ES = Elasticsearch(es_server, verify_certs=False,use_ssl = True, http_auth= "admin:admin")
  
  query = json.loads(es_query)
  
  doc_counts  = {}
  missings = {}
  
  for c in classes:
    
    logger.info("Loading documents from `{}` for class `{}`".format(es_index,c))
  
    doc_counts[c] = 0
    missings[c] = 0
    
    
    query["query"]["bool"]["must"][0]["query_string"]["default_field"] = doc_ids
    query["query"]["bool"]["must"][0]["query_string"]["query"] = c
        
    results_gen = helpers.scan(ES, query = query, index=es_index, scroll = '30s')
    
    for i,doc in enumerate(results_gen):
      
      try:
        
        # keep source
        doc = doc['_source']
        
        doc_counts[c] += 1
        
        # change dict keys for compatibility with gensim data structure (TaggedDocument)
        
        labels = [x for x in doc.pop(doc_ids) if not x.startswith(c.replace('*','')) ]
        
        labels = labels if labels else 'UNK'
        
        doc['tags'] = ['\t'.join([c,labels[0],str(i)])]
        doc['words'] = doc.pop(doc_texts)
        
        if doc['words'] != None:
          
          yield doc
          
        else:
          missings[c] += 1
      
      except KeyError:
        missings[c] += 1
    
      _show_loading_buffer(i)
  
    logger.info("\nDocuments for class `{}` : {} - missing {}".format(c,doc_counts[c], missings[c])) 
    
        
def load_json(folder,doc_id,doc_text):
  """
  Generator of documents from JSON files. Recursion is enabled. Fields of intereset must be consistent across all JSON files.
  
  :params:
    folder (str) : root of path where JSON files are stored
    doc_id (str) : filed to be used as id for document
    doc_text (str) : filed where text to be processed is stored
    
  :return:
    generator

  """
  
  doc_counts = {}
  
  files = [file for file in glob.glob(folder + '/**/*.json', recursive=True)]
  
  logger.info("Loading documents from {0}".format(folder))
  
  file_origins = [filename.split(os.sep)[-2] for filename in files]
  
  for file_orig in file_origins:
    doc_counts[file_orig] = 0
  
  missing_field = 0
  
  for filename,file_orig in zip(files,file_origins):
    
    with open(filename) as infile:
              
      for idx,line in enumerate(infile):
        json_line = json.loads(line)
                  
        doc_counts[file_orig] += 1
        
        # CHANGE DICT KEYS FOR GENSIM COMPATIBILITY
        json_line['words'] = json_line.pop(doc_text)
        
        json_line['tags'] = ['\t'.join([file_orig,json_line.pop(doc_id)])] 
          
        yield json_line
                          
        _show_loading_buffer(idx)
        
    logger.info("Documents in {} discarded beacuse of missing field : {}".format(filename,missing_field))

  _new_logging_block()
  
  return doc_counts
        
   
def load_data_from_config(config_load):
  """
  Wrapper function to retrieve data with method specified in configuration file.
  
  :param conf: configuration file
  :type conf: configparser.ConfigParser
  
  :rtype: generator
  """
  load = config_load.get('load')
  doc_ids = config_load.get('doc_ids')
  doc_texts = config_load.get('doc_texts')
  
  
  if not load:
    
    logger.warning("Loading method undefined!")

  if not doc_ids and not doc_texts:
    
    logger.warning("You need to specify the fields to be used as id and text for documents!")
    
  if load.startswith('es'):
    
    es_server = config_load.get('es_server','localhost')
    es_indices = config_load.get('es_indices')
    es_query = config_load.get('es_query','{"query" : {"match_all" : {}}}')
    
    if not es_server:
      
      logger.warning("You need to specify a location for ES instance!")
    
    if not es_indices:
      
      logger.warning("You need to specify an index!")
      
    if load == 'es_std':
    
      es_indices = es_indices.split(',')
      doc_ids = doc_ids.split(',')
      doc_texts = doc_texts.split(',')
      
      data = load_es_by_index(es_server,es_indices,es_query,doc_ids,doc_texts)
      
    elif load == 'es_class':
      
      classes = config_load.get('doc_classes').split(',')
      
      data = load_es_by_class(es_server,es_indices,es_query,doc_ids,doc_texts,classes = classes)
  
  elif load == 'json':
    
    folder = config_load.get('json_path')
    
    if not folder:
      logger.warning("You need to specify the folder where JSON files are stored!")
    
    data = load_json(folder,doc_ids,doc_texts)
    
  return data
  
  
 
#def load_xml(folder,doc_id,doc_text):
#  """
#  Generator of documents from XML files. Recursion is enabled. 
#  
#  :param folder: root of path where XML files are stored
#  :param doc_id: filed to be used as id for document. MUST BE AN ATTRIBUTE OF THE ROOT TAG e.g. :
#    
#    <root doc_id=doc_id> rest of xml
#  
#  :param doc_text: filed where text to be processed is stored. MUST BE A TAG CHILD OF ROOT e.g.
#    <root > 
#      <doc_text> 
#  
#  """
#  files = [file for file in glob.glob(folder + '/**/*.xml', recursive=True)]
#    
#  for filename in files:
#    tree = ET.parse(filename)
#    root = tree.getroot()
#    elem_text = root.find(doc_text)
#    
#    if elem_text != None:
#    
#      text = [t for t in elem_text.itertext()]
#      
#    else:
#      
#      logger.debug("Tag not found in doc : {}".format(filename))
#    
#    yield {'id' : root.attrib.get(doc_id), 'text' : ''.join(text)}
      
 
class ModelsLoader:
  """
  Class for loading or constructing`gensim.models`. Load either models from file/directory 
  or new construct model instances defined by a parameter grid.
  """
  
  @classmethod
  def _serial_val(cls,key,values):
    """
    Serialize values of parameter grid items.
    
    :param key: model parameter
    :type key: str
    :param values: values for parameter
    :type values: list of int/float
    
    :rtype: list
    """
    if key == 'sample':
      values = [float(x) for x in values.split(',')]
    else:
      values = [int(x) for x in values.split(',')]
    
    return values
  
  @classmethod
  def _print_msg(cls,models):
    """
    Print name of models created/loaded
    """
    
    logger.info("Loaded : {} models".format(len(models)))
    
  @staticmethod
  def load_from_dir(models_dir):
    """
    Load all models present in a directory.
    
    :param models_dir: path to directory where models are stored
    :type models_dir: str
    
    :rtype: list 
    """
    
    models = [doc2vec.Doc2Vec.load(model) for model in glob.glob(os.path.join(models_dir,'*')) if not model.endswith('.npy')]
    
    ModelsLoader._print_msg(models)
      
    return models
  
  @staticmethod
  def load_single_model(model_path):
    """
    Load single model from file.
    
    :param model_path: path to model
    :type model_path: str
    
    :rtype: list 
    """
    
    models = [doc2vec.Doc2Vec.load(model_path)]
    
    ModelsLoader._print_msg(models)
    
    return models
    
  @staticmethod
  def construct_models(parameters_conf):
    """
    Construct new models from dictionary (parameters,values) specified in configuration file. Dictionary is converted to 
    `sklearn.model_selection import ParameterGrid` for parameters combination.
    
    :param parameters_conf: parameters for models
    :type param_conf: configparser.ConfigParser
    
    :rtype: list
    """
    
    dict_conf = {k : ModelsLoader._serial_val(k,v) for k,v in dict(parameters_conf.items()).items()}  
    param_grid = ParameterGrid(dict_conf)
    
    
    models_dict = {}
    for param_conf in param_grid:
      model = doc2vec.Doc2Vec(dm= param_conf.get('dm'),
                              window = param_conf.get('window'),
                              size = param_conf.get('size'),
                              hs = param_conf.get('hs'),
                              min_count = param_conf.get('min_count'),
                              dm_concat = param_conf.get('dm_concat'),
                              dm_mean = param_conf.get('dm_mean'),
                              negative = param_conf.get('negative'),
                              workers = param_conf.get('workers',cpu_count()),
                              sample = param_conf.get('sample')
                              )
      
      models_dict[str(model)] = model
      
    models = list(models_dict.values())  
    ModelsLoader._print_msg(models)
    
    return models
  
  @classmethod
  def load_models_from_config(cls,load_config,param_config):
    """
    Wrapper function to load models as specified in the configuration file. 
    
    :param load_config: configuration file section for loading
    :type load_config: configparser.ConfigParser
    
    :param param_config: configuration file section for models' parameters. Will be ignored if no new models are created
    :type param_config: configparser.ConfigParser
    
    :rtype: list of models

    """
    
    if load_config.getboolean('constuct_new_models',False):
      
      models = cls.construct_models(param_config)
    
    elif load_config.get('load_all_models'):
      
      models = cls.load_from_dir(load_config.get('load_all_models'))
      
    elif load_config.get('load_single_model'):
    
      models = cls.load_single_model(load_config.get('load_single_model'))
      
    return models
    
      
    
@timeit
def load_and_process(doc_gen):
  """
  Listify generator in order to actualize processes.
  
  :param doc_gen: generator of documents
  :type doc_gen: generator
  
  :rtype: list
  """
  all_docs =  list(doc_gen)
  logger.info("Retrieved {} documents".format(len(all_docs)))
  return all_docs
    
    
def load_nih_ground_truth(infile):
  """
  Load ground truth labels for `nih` patent-paper recommendation system. 
  File must beline separated in the form : `nih-patent-<ID>` \t nih-paper-<ID>` 
  
  :param infile: file where groud truth labels are stored
  :rtype: dict with recommendations patent -> papers 
  """

  gt_ptpp = defaultdict(list)
  
  with open(infile) as f:
    for line in f:
      patent_id,paper_id = line.split('\t')
      gt_ptpp[patent_id].append(paper_id.strip())
  
  return gt_ptpp
  
