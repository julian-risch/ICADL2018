#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:55:31 2018

@author: Samuele Garda
"""

import argparse
import logging
import pickle
import json
import numpy as np
import pandas as pd
from create_exp_data import create_dir,join_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors 
import random
from semsim_score import sym_sent_sim
import matplotlib.pyplot as plt



logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO') 


def parse_args():
  
  parser = argparse.ArgumentParser(description='Experiment with location in document embeddings')
  parser.add_argument("--data-path",required = True, help = "Path to dir to JSON with parsed input")
  parser.add_argument("--vocab-path",required = True, help = "Path to dir to folder with vocabulary files")
  parser.add_argument("--embd-path",required = True, help = "Path to dir to folder with emeddings matrices")
  parser.add_argument("--top-k", type = int, default = 10, help = "Top `k` most similar")
  parser.add_argument("--exp-fixothers",action = 'store_true', help = "Perform experiment with location change")
  parser.add_argument("--exp-viz", default = False, help = "Write data for visualization of embedding space")
  parser.add_argument("--exp-mix", action = "store_true", help = "Mix plot embeddings")
  parser.add_argument("--exp-plot", default = 0, type = int, help = "Plot BoW vs W2V w.r.t. overlap score")
  parser.add_argument("--evaluate", default = 0, type = int, help = "Evaluate method against BoW baseline. Set to 0 for skip.")
  parser.add_argument("--loc", default = "germany",type = str, help = "Add new location to plot. Active only within `--exp-fixothers`")
  return parser.parse_args()


#def write_embd_viz(vocab,w2i,out_file):
#
#    for word in vocab:
#      out_metadata.write("loc\t{}\n".format(word))
#      vector = '\t'.join(str(x) for x in locs_emb_matrix[w2i[word]])
#      out_emb.write("{}\n".format(vector))


def get_overlap_score(book, single_recom):
  """
  Compute vocabulary overlapping score.
  
  :math:`\\frac{V_{book} \\cap V_{rec}}{V_{book}}`
  
  :params:
    book (list) : list of tokens
    single_recom (list) : list of tokens
  
  :return:
    overalp_score (int) : weighted amount of overlapping words
  
  """
  
  overlap_score = len(set(book).intersection(set(single_recom)))/len(set(book))
  
  return overlap_score

def get_overlap(data,test_idx_data,top_k_idx,mode = 'threshold', t = 20):
  """
  Finde threshold for mixed system (BoW + W2V).
  
  :params:
    data (pandas.Dataframe) : dataset
    test_idx_data (list) : list of indices (test cases)
    top_k_idx (list) : list of list of recomendations
    t (int) : nth overlapping score to be used as threshold
    
  :return:
    threshold (int) : overlapping score to be used to decide for BoW or W2V vectors
  """
  
  scores = []
  
  for idx,test_idx in enumerate(test_idx_data):
    
    ols = [get_overlap_score(data['nonloc_tok'][test_idx],data['nonloc_tok'][top_k_idx[idx][i]]) for i in range(len(top_k_idx[idx]))]
    scores.append(ols)
    
  sorted_scores = sorted([score for sublist in scores for score in sublist])
    
  if mode == 'threshold':
    
    return sorted_scores[t]
    
  elif mode == 'scores':
    
    return sorted_scores


def get_mixed_model(bow_top_k_idx, w2v_top_k_idx, idx_test_data,threshold, k):
  """
  Create recomendations with mixed models. If overalp score is below threshold W2V model is used else BoW.
  
  :params:
    bow_top_k_idx (list) : list of list of recomendations (n recomendations for each test case)
    w2v_top_k_idx (list) : list of list of recomendations (n recomendations for each test case)
    idx_test_data (list) : list of test cases (indices)
    threshold (int) : if overlap score below this value W2V model is used
    k (int) : # of recomendations
  
  :return:
    mixed_top_k_idx (list) : list of list of recomendations
    
  """
  
  mixed_top_k_idx = []
      
  for idx,test_idx in enumerate(idx_test_data):
    single_rec_idx = []
    for i in range(k):
      ol_score = get_overlap_score(data['nonloc_tok'][test_idx],data['nonloc_tok'][bow_top_k_idx[idx][i]])
      if ol_score <= threshold:
        rec = w2v_top_k_idx[idx][0]
      else:
        rec = bow_top_k_idx[idx][i]
      single_rec_idx.append(rec)
    mixed_top_k_idx.append(single_rec_idx)
  
  return mixed_top_k_idx

def cosine_similarity(vector,matrix):
  """
  Compute cosine similarity of vector with each vector belonging to matrix.
  
  :math:`cos(x,y)=\\frac{x \cdot y}{||x|| \cdot ||y||}`
  
  :params:
    vector (np.ndarray) : vector
    matrix (np.ndarray) : matrix
    
  :return:
    similarites (np.ndarray) : vector containing cosine similarities \
    of input vector with each row in matrix
  """
  
  
  norm = np.linalg.norm(vector)
  all_norms = np.linalg.norm(matrix, axis=1)
  dot_products = np.dot(matrix, vector)
  similarities = dot_products / (norm * all_norms)
  
  return similarities


def load_json_data(data_path):
  """
  Load JSON file (one item per line)
  
  :params:
    data_path (str) : path to JSON file
    
  :return:
    
    data (pandas.DataFrame) : DataFrame with columns being JSON fields
  """
  
  data = []
  
  with open(data_path) as infile:
    for line in infile:
      file = json.loads(line)
      data.append(file)
      
  return pd.DataFrame(data)


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


def load_pickle(file_path):
  """
  Load item saved in pickle file.
  
  :params:
    file_path (str) : path to pickle file
  :return:
    item : content of file
    
  """
  item = pickle.load(open(file_path, mode = "rb"))
  return item


def single_book_qual_anal(data,book_idx,rec_indices, pred):
  rec_indices = rec_indices[1:]
    
  recomnedations = [{rec_idx : [data.title[rec_idx], data.genre[rec_idx]]} for rec_idx in rec_indices]
  logger.info("Recomendations:\n{}".format(recomnedations))
  
  rec_simscore = [sym_sent_sim(data['nonloc_tok'][book_idx],data['nonloc_tok'][rec_idx]) for rec_idx in rec_indices]
#  logger.info("First hit score : {}".format(rec_simscore[0]))
  logger.info("Average simscore : {}".format(np.mean(rec_simscore)))
  

if __name__ == "__main__":
  
  random.seed(42)
  np.random.seed(42)
  
  args = parse_args()

  TEST_BOOK_FIXOTHER  =  ["The Twilight of the Idols"]
      
#      "Nemesis","The Thirteen Problems","Partners in Crime","Giant's Bread"]
  
  #["The Mysterious Affair at Styles"]
                          
#                          "The Twilight of the Idols",
  
#  ["Twilight"]
  
#  ["A Clockwork Orange", "Don Quixote","Dracula","Hamlet","Ivanhoe","Moby-Dick; or, The Whale","Odyssey","The Shining","The Hound of the Baskervilles",
#                         "Adventures of Huckleberry Finn","The Adventures of Tom Sawyer","Pride and Prejudice","War and Peace","The Sorrows of Young Werther",
#                         "Where the Red Fern Grows","Robinson Crusoe","Doktor Faustus","One Hundred Years of Solitude","Oliver Twist","Nineteen Eighty-Four"]
#["Harry Potter and the Deathly Hallows","Harry Potter and the Order of the Phoenix"] 
#["The Great Train Robbery", "Country of the Blind", "Dragon's Eye: A Chinese Noir","The Colorado Kid"] 
#["The Charioteer of Delphi", "The Dragon Lord", "The Whiskey Rebels","The Gladiators from Capua"]
  
  TEST_LOC = args.loc
  
  logger.info("Loading processed summaries from `{}`".format(args.data_path))
  
  data = load_json_data(args.data_path)
  
  logger.info("Loading vocabulary files from `{}`".format(args.vocab_path))
  
  locs_vocab = load_pickle(join_path([args.vocab_path, "loc.vocab"]))
  
  other_vocab = load_pickle(join_path([args.vocab_path, "non-loc.vocab"]))
  
  other_w2i = load_pickle(join_path([args.vocab_path, "non-loc.w2i"]))
  
  locs_w2i = load_pickle(join_path([args.vocab_path, "loc.w2i"]))
    
  logger.info("Loading embedding matrices from `{}`".format(args.embd_path))
  
  locs_emb_matrix = np.load(join_path([args.embd_path, "loc.npy"]))
  
  oth_emb_matrix = np.load(join_path([args.embd_path, "non-loc.npy"]))
  
  docs_others = list(data.nonloc_tok)
  
  docs_locs = list(data.loc_tok)
  
  X_others = [sent_to_indices(sent,other_w2i) for sent in docs_others]
  
  X_locs = [sent_to_indices(sent,locs_w2i) for sent in docs_locs]
  
  if args.exp_viz:
  
    create_dir(args.viz)
    
    logger.info("Creating files for visualization at `{}`".format(args.viz))
    
    out_emb = open("{}.vectors".format(join_path([args.viz,args.viz])), "w")
    out_metadata = open("{}.metadata".format(join_path([args.viz,args.viz])), "w")
    
    out_metadata.write("type\tname\n")
  
    for loc in locs_vocab:
      out_metadata.write("loc\t{}\n".format(loc))
      vector = '\t'.join(str(x) for x in locs_emb_matrix[locs_w2i[loc]])
      out_emb.write("{}\n".format(vector))
      
    for other in other_vocab:
      out_metadata.write("other\t{}\n".format(other))
      vector = '\t'.join(str(x) for x in oth_emb_matrix[other_w2i[other]])
      out_emb.write("{}\n".format(vector))

  logger.info("Processing book summaries embeddings...")
  
  Ws_locs = []
  
  Ws_others = []
  
  to_drop = []

  for idx,(X_loc,X_other) in enumerate(zip(X_locs,X_others)):
    
    if X_loc and X_other:
      
      # create location embedding
      W_locs = np.mean(locs_emb_matrix[X_loc],axis = 0)

      # create other embeddings
      W_others = np.mean(oth_emb_matrix[X_other], axis = 0)
      
      Ws_locs.append(W_locs[np.newaxis,:])
        
      Ws_others.append(W_others[np.newaxis,:])
      
      if args.exp_viz:
      
#        write embeddings of book without location
        out_metadata.write("non-loc-book\t{}\n".format(data.title[idx]+'_no_loc'))
              
        out_emb.write("{}\n".format('\t'.join(str(x) for x in W_others)))
      
        D = np.sum([W_locs,W_others], axis = 0)
        
        # write embedding of book with location
        out_metadata.write("book\t{}\n".format(data.title[idx]))
              
        out_emb.write("{}\n".format('\t'.join(str(x) for x in D)))
        
        if (idx+1) % 1000 == 0:
          logger.info("Created {} document embeddings".format(idx))
          
          
        if data.title[idx] in TEST_BOOK_FIXOTHER:
          
          test_new_book = np.sum([W_others, locs_emb_matrix[locs_w2i[TEST_LOC]]], axis = 0)
                                
          out_metadata.write("immaginary-book\t{}_in_{}\n".format(data.title[idx],TEST_LOC))
          
          out_emb.write("{}\n".format('\t'.join(str(x) for x in test_new_book)))   
            
      
    else:
      
      to_drop.append(idx)
      
  
  data = data.drop(data.index[[to_drop]])
    
  data.index = range(len(data))
  
  k = args.top_k + 1
          
  
  if args.evaluate != 0:
    
    logger.info("Start evaluation process on {} cases".format(args.evaluate))
        
    idx_test_data = np.random.randint(low = 0, high = len(data), size = args.evaluate)
        
    vect = TfidfVectorizer(preprocessor = ' '.join, tokenizer = str.split, ngram_range = (1,1), use_idf = False)
  
    texts = list(data['nonloc_tok'])

    X = vect.fit_transform(texts)
    
    clf = NearestNeighbors(n_neighbors=k)
    
    logger.info("Fitting KNN to non location data")
    
    clf.fit(X)
    
    logger.info("Find `{}` recommendation with KNN".format(k))
    
    top_k_idx_bow = [clf.kneighbors(X[idx_test],return_distance = False)[0] for idx_test in idx_test_data]

    Ws_others = np.concatenate(Ws_others, axis = 0)
  
    logger.info("Find `{}` recommendation with W2V".format(k))
    
    sims_noloc = [cosine_similarity(Ws_others[idx_test],Ws_others) for idx_test in idx_test_data]
      
    top_k_idx_w2v =  [np.argpartition(-sim_noloc.ravel(),range(k), axis = 0)[:k] for sim_noloc in sims_noloc]
    
    overlap_threshold = get_overlap(data,idx_test_data,top_k_idx_bow,mode = 'threshodl', t = 20)
    
    logger.info("Overlapping threshold for Mixed model is {}".format(overlap_threshold))
    
    top_k_idx_mixed = get_mixed_model(top_k_idx_bow, top_k_idx_w2v, idx_test_data,overlap_threshold, k)
    
    logger.info("Computing WordNet Semantic Similarity Score")
    
    first_hit_scores_w2i = []
    first_hit_scores_bow = []
    first_hit_scores_mixed = []
    average_10_hits_w2i = []
    average_10_hits_bow = []
    average_10_hits_mixed = []
    
    for idx,test_idx in enumerate(idx_test_data):
      
      w2i_scores = [sym_sent_sim(data['nonloc_tok'][test_idx],data['nonloc_tok'][top_k_idx_w2v[idx][i]]) for i in range(1,k)]
            
      bow_scores =  [sym_sent_sim(data['nonloc_tok'][test_idx],data['nonloc_tok'][top_k_idx_bow[idx][i]]) for i in range(1,k)]
      
      mixed_scores = [sym_sent_sim(data['nonloc_tok'][test_idx],data['nonloc_tok'][top_k_idx_mixed[idx][i]]) for i in range(1,k)]
      
      if (idx+1) % 10 == 0:
        logger.info("Processed {} test case".format(idx+1))
      
      logger.info("W2V : {}".format(w2i_scores))
      logger.info("BoW : {}".format(bow_scores))
            
      first_hit_scores_w2i.append(w2i_scores[0])
      first_hit_scores_bow.append(bow_scores[0])
      first_hit_scores_mixed.append(mixed_scores[0])
      average_10_hits_w2i.append(np.mean(w2i_scores))
      average_10_hits_bow.append(np.mean(bow_scores))
      average_10_hits_mixed.append(np.mean(mixed_scores))
      
    
    logger.info("Average first hit similarity score for W2V on {} book summaries : {}".format(args.evaluate, np.mean(first_hit_scores_w2i)))
    logger.info("Average first hit similarity score for BoW on {} book summaries : {}".format(args.evaluate, np.mean(first_hit_scores_bow)))
    logger.info("Average first hit similarity score for Mixed on {} book summaries : {}".format(args.evaluate, np.mean(first_hit_scores_mixed)))
    logger.info("Average 10 hits similarity score for W2V on {} book summaries : {}".format(args.evaluate, np.mean(average_10_hits_w2i)))
    logger.info("Average 10 hits similarity score for BoW on {} book summaries : {}".format(args.evaluate, np.mean(average_10_hits_bow)))
    logger.info("Average 10 hits similarity score for Mixed on {} book summaries : {}".format(args.evaluate, np.mean(average_10_hits_mixed)))
    
    
  elif args.exp_plot:
    
    idx_test_data = np.random.randint(low = 0, high = len(data), size = args.exp_plot)
    
    vect = TfidfVectorizer(preprocessor = ' '.join, tokenizer = str.split, ngram_range = (1,1), use_idf = False)
  
    texts = list(data['nonloc_tok'])

    X = vect.fit_transform(texts)
    
    clf = NearestNeighbors(n_neighbors=k)
    
    logger.info("Fitting KNN to non location data")
    
    clf.fit(X)
    
    logger.info("Find `{}` recommendation with KNN".format(k))
    
    top_k_idx_bow = [clf.kneighbors(X[idx_test],return_distance = False)[0] for idx_test in idx_test_data]

    Ws_others = np.concatenate(Ws_others, axis = 0)
  
    logger.info("Find `{}` recommendation with W2V".format(k))
    
    sims_noloc = [cosine_similarity(Ws_others[idx_test],Ws_others) for idx_test in idx_test_data]
      
    top_k_idx_w2v =  [np.argpartition(-sim_noloc.ravel(),range(k), axis = 0)[:k] for sim_noloc in sims_noloc]
    
    logger.info("Starting creating scores for plot")
  
    
    simscore_table = {}
    df_index = 0
    for idx,test_idx in enumerate(idx_test_data):
      logger.info("Processing test case : {}".format(idx+1))
      for i in range(1,k):
        simscore_table[df_index] = {}
        simscore_table[df_index]['BoW'] = sym_sent_sim(data['nonloc_tok'][test_idx],data['nonloc_tok'][top_k_idx_bow[idx][i]])
        simscore_table[df_index]['W2V'] = sym_sent_sim(data['nonloc_tok'][test_idx],data['nonloc_tok'][top_k_idx_w2v[idx][i]])
        simscore_table[df_index]['overlap'] = get_overlap_score(data['nonloc_tok'][test_idx],data['nonloc_tok'][top_k_idx_bow[idx][i]])
        df_index += 1
        
    data_score = pd.DataFrame.from_dict(simscore_table, orient='index')
    
    pickle.dump(simscore_table, open('./score_table.dict', 'wb'))
    
    data_score = data_score.sort_values(by = 'overlap')
    
    ax = data_score.plot(x = 'overlap')
    
    fig = ax.get_figure()
    fig.savefig('./scores_plot.50t.10k')
    
    
    ax = data_score.plot(kind="scatter", x = 'overlap',y= 'BoW', color="b", label="BoW")
    data_score.plot(x="overlap",y="W2V", color="r", label="W2V", ax=ax, kind = 'scatter')
    ax.set(xlabel = 'overlap', ylabel = 'simscore')
    plt.savefig('./scores_scatter')  
    
        
  elif args.exp_fixothers:
    
    logger.info("Evaluating location embeddings arithmetics with : {}".format(args.loc))
    
    logger.info("Creating W2V models")
    
    Ws_locs = np.concatenate(Ws_locs, axis = 0)
    
    Ws_others = np.concatenate(Ws_others, axis = 0)
    
    D = np.sum([Ws_locs,Ws_others],axis = 0)
    
    new_location = locs_emb_matrix[locs_w2i[TEST_LOC]]
    
    book_to_test = TEST_BOOK_FIXOTHER 
    
    logger.info("Creating BoW baselines...")
    
    vect = TfidfVectorizer(preprocessor = ' '.join, tokenizer = str.split, ngram_range = (1,1), use_idf = False)
  
    texts_noloc = list(data['nonloc_tok'])
    X_noloc = vect.fit_transform(texts_noloc)
    clf_noloc = NearestNeighbors(n_neighbors=k)
    clf_noloc.fit(X_noloc)
    
    texts_loc = [nonloc_tok + loc_tok for nonloc_tok,loc_tok in zip(data['nonloc_tok'],data['loc_tok'])]
    X_loc = vect.fit_transform(texts_loc)        
    clf_loc = NearestNeighbors(n_neighbors=k)
    clf_loc.fit(X_loc)
    
    logger.info("Start qualitative evaluation process...")
    
    for book in book_to_test:
      
      book_idx = data.index[data.title==book][0]
      
      logger.info("Assessing recommendations for:\n\n{} - {} - {}".format(data.title[book_idx], data.genre[book_idx],
            data.loc_tok[book_idx]))
#      print("-"*50 + '\n')
      
#       BoW recomendatoins indices
      bow_noloc_idx = clf_noloc.kneighbors(X_noloc[book_idx],return_distance = False)[0]
      
      bow_loc_idx = clf_loc.kneighbors(X_loc[book_idx],return_distance = False)[0]
            
#       W2V recomendations indices
      sims_noloc = cosine_similarity(Ws_others[book_idx],Ws_others)
      
      w2v_noloc_idx =  np.argpartition(-sims_noloc.ravel(),range(k), axis = 0)[:k]
      
#      alphas = [round(x * .01,1) for x in range(10,100,10)] + ['mean']
#      alphas = [0.2,0.3,0.4,1]
#      
#      for alpha in alphas:
#        
#        logger.info("Normalize location with alpha = {}\n".format(alpha))
#        
#        if isinstance(alpha,float):
#        
#          norm_new_loc = new_location * alpha
#        
#          test_new = np.sum([Ws_others[book_idx], norm_new_loc], axis = 0) 
#          
#        else:
#          
#          test_new = np.mean([Ws_others[book_idx], new_location], axis = 0)
#          
#        sims_new_D = cosine_similarity(test_new,D)
#        
#        w2v_new_D =  np.argpartition(-sims_new_D.ravel(),range(k), axis = 0)[:k]
#        
#        logger.info("\nW2V_loc_D:\n")
#        recomnedations = [{rec_idx : [data.title[rec_idx], data.genre[rec_idx], data.loc_tok[rec_idx]]} for rec_idx in w2v_new_D]
#        logger.info("Recomendations:{}\n\n\n".format(recomnedations))
        
        
        
        
#       New Location + W2V recomendations indices
      test_new = np.sum([Ws_others[book_idx], new_location], axis = 0)
      
      sims_new_D = cosine_similarity(test_new,D)
      
      w2v_new_D =  np.argpartition(-sims_new_D.ravel(),range(k), axis = 0)[:k]
      
      sims_new_Ws_others = cosine_similarity(test_new,Ws_others)
      
      w2v_new_Ws_others =  np.argpartition(-sims_new_Ws_others.ravel(),range(k), axis = 0)[:k]
      
      
      logger.info("\nW2V_loc_D:\n")
      recomnedations = [{rec_idx : [data.title[rec_idx], data.genre[rec_idx]]} for rec_idx in w2v_new_D]
      logger.info("Recomendations:\n{}".format(recomnedations))
     
      logger.info("\nBow_noloc:\n")
      single_book_qual_anal(data,book_idx,bow_noloc_idx, pred = 'bow')
##      logger.info("\nBow_loc:\n")
##      single_book_qual_anal(data,book_idx,bow_loc_idx, pred = 'bow')
      logger.info("\nW2v_noloc:\n")
      single_book_qual_anal(data,book_idx,w2v_noloc_idx,pred = 'w2v')
#      
#      logger.info("\nW2V_loc_D:\n")
#      recomnedations = [{rec_idx : [data.title[rec_idx], data.genre[rec_idx]]} for rec_idx in w2v_new_D]
#      logger.info("Recomendations:\n{}".format(recomnedations))
      
#      logger.info("\nW2V_loc_Ws_others:\n")
#      single_book_qual_anal(data,book_idx,w2v_new_Ws_others, pred = 'w2v')
#    
      
#      logger.info('---------------------------------------------------------------\n')
      
  elif args.exp_mix:
    
    logger.info("Evaluating mixture of plot...")
    
    Ws_others = np.concatenate(Ws_others, axis = 0)
    
    romance_idx = [idx for idx,gen in enumerate(list(data.genre)) if  isinstance(gen,dict) and "Romance novel" in gen.values()]
        
    crime_idx = [idx for idx,gen in enumerate(list(data.genre)) if  isinstance(gen,dict) and "Crime Fiction" in gen.values()]
        
    romance_samples = random.sample(romance_idx, 10)
    
    crime_samples = random.sample(crime_idx, 10)
    
    for rom_s,cri_s in zip(romance_samples,crime_samples):
      
      logger.info("Assessing : {} - {} +\n".format(data.title[rom_s],data.genre[rom_s]))
      logger.info("{} - {}\n".format(data.title[cri_s],data.genre[cri_s]))
      
      if rom_s != cri_s:
      
        romance_plt = Ws_others[rom_s]
        crime_plt = Ws_others[cri_s]
        
        avg = np.mean([romance_plt,crime_plt], axis = 0)
        
        sims_Ws_others = cosine_similarity(avg,Ws_others)
        
        w2v_Ws_others =  np.argpartition(-sims_Ws_others.ravel(),range(k), axis = 0)[:k]
        
        recomnedations = [{rec_idx : [data.title[rec_idx], data.genre[rec_idx]]} for rec_idx in w2v_Ws_others]
        
        logger.info("Recomendations:\n{}".format(recomnedations))
        print("\n")
      
    else:
      pass
      
      
      
      
      
      
      
    
  
  
    
    
      
      
      

      
      
      
      
      
      
      
      
      
     
      
      
      
      
      
