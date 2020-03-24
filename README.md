## Citation

If you use our work, please cite our paper [**Book Recommendation Beyond the Usual Suspects: Embedding Book Plots Together with Place and Time Information**](https://github.com/julian-risch/ICADL2018/risch2018book.pdf) as follows:

    @inproceedings{risch2018book,
    author = {Risch, Julian and Garda, Samuele and Krestel, Ralf},
    booktitle = {Proceedings of the International Conference On Asia-Pacific Digital Libraries (ICADL)},
    pages = {227-239},
    title = {Book Recommendation Beyond the Usual Suspects: Embedding Book Plots Together with Place and Time Information},
    year = {2018}
    }


# doc-embedding
Python scripts to train doc2vec models. For each step of the pipeline please refer to the correspondent sections in the configuration files in `configs`. To train models you can call:

    $ python train_models.py -c ./configs/example.config

## CONFIGURATION FILE

### LOADING

This section of the configuration file determines the specification in the input step of the pipeline (related to the script `load.py`)
First of all you need to specify whether the documents you want to train your model on are stored in JSON files or in an Elasticsearch instance. Now only these two options are allowed : `es`,`json`

    load = es

If you are loading json files you need to provide a path to a folder storing them. You can have as many subfolders as you want since the loading is recursive.

    json_path = /home/users/me/json_data
    
If you are loading data from ES then you need to specify where the ES instance is hosted and the name of the indices. Moreover since it is possible to load data from multiple indices you need a `query`: be carefull this will be applied to each of the indices 
so make sure to make it as general as possible in order to avoid fields mismatching for instance. Finally you need the document fields that will point to the absolutely necessery attributes of gensim data structure : words and tags. Again be carfel: be sure to place them in the corresponding order of the ES indices in order to avoid mismatching fields names

    es_indicis = index_1,index_2,index_3
    es_query = {"query" : {"match_all" : {}}, "sort" : ["_doc"]} # to be completely sure, let's grab everything
    doc_ids = tag_of_doc_in_index_1,tag_of_doc_in_index_2,tag_of_doc_in_index_3
    doc_texts = words_of_doc_in_index_1,words_of_doc_in_index_2,words_of_doc_in_index_3
    
    
Now that the data is loaded we can take care of the models. If you already have trained some model and want to further improve them go for:

    load_all_models = ./models
    load_single_model = ./models/Doc2Vec(dm-m,d100,n20,w5)
    
The first one will make the script load all the models in that folder, the second one only the specified one (pretty clear huh?).
If instead you have to start from scratch please set `constuct_new_models = True`. Please refer to the section `PARAMETERS` for further instructions.

### PREPROCESS

Here is were you can specify which type of preprocessing you want to apply to your data. So far you can only choose if you wish to remove stopwords (kindly provided by `gensim`) and or stem the text. If none of the before is set to true the documents will be simply tokenized and any trace of HTML tags remove. Look here for more information: https://radimrehurek.com/gensim/parsing/preprocessing.html

    rm_stopwords
    stem

### PARAMETERS

If in the `LOADING` phase you set `constuct_new_models = True` you'll end up here. In this section all the parameters for your brand new models will be defined. Since this could easily take some pages to describe what all the options are doing please referer
directly to gensim documentation ( https://radimrehurek.com/gensim/models/doc2vec.html ) and to have a full understanding of the model read the original papers from [Mikolov et. al](#references)
Nonetheless what you need to know is that for each parameter you can specify as many values as you wich - COMMA SEPARATED - that will be used then to instatiate models with all the possible configuration you specified.

### TRAINING

Finally on the training part. Here there's an important decision to make. If you set 

    adapt_alpha = False
    
the models will be trained with the standard gensim function. This mean that the learning rate will be adapted during training as specified by gensim.doc2vec developers. If instead it is set to `True` the learning rate decay will follow this rule `alpha_delta = (alpha - min_alpha) / epochs` as specified in this jupyter notebook https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb where gensim developers tried to reproduce [Mikolov et. al](#references) results.
Please take a look at `train_models.ModelsTrainer.train_manual_lr` if you wish to set a different approach for the learning rate decay.
If `adapt_alpha = True` parameters:
    
    alpha
    min_alpha
    
will be used as starting value and minimal value to be reached at the end of the training process for the learning rate. Otherwise will be ignored.
Moreover you can specify:
    
    epochs
    checkpoint
    shuffle
    
i.e. the number of epochs, the folder where the model will be stored and whether to shuffle the documents for training. Please note that depending on the value of `adapt_alpha` the training behaviour might slightly change. I.e. if the gensim standard train function is used all the training process is done in one function call, thus the documents will be shuffled only once before training and the models will not be accessible before the training procedure is terminated (and no logging during this phase), otherwise the documents will be shuffled at each epoch with some logging for the learning rate value, etc.
If your are using the trained embeddings in a classification task (e.g. the newspaper one) you might want to downsample your dataset due to the high present of a category. This can be achieved by
    
    downsample
    
All the documents of categories that have more samples than the least frequent will be pruned. 

Finally at the very end of the training process it is possible to have a quick look at the quality of your embeddings thaks to:
    
    quality_check_infered = 5
    quality_check_trained = 5
    
For the first option a random document will be selected from the corpus and a new vector will be inferred. Then the first `n` most similar documents will be print out. This is to check whether your model can infer proper vector assuming it never saw such a document. For this test, if the training process was successfull, the most similar document should be the same randomly picked. 
The second one instead will make the script print out the `n` most similar training vectors for a random document. In this way you can check by yourself if the documents are related or note

### LABELS

Coming soon...

#### Important

If you wish to disable any of the options above mentioned you just need to comment it out or leave it empty.

## EMBEDDINGS PROJECTOR

Script `embeddings_projector` allows you to visualize the embeddings produced after the training process.
    
    -d, --dir 
    -o, --out-dir 
    -n, --n-docs 
    -f, --fields

You need to specify a folder where all your trained model are stored. After the models are loaded in the folder passed to `-o` will be created as many folders as models. For each folder (corresponding to a model) you can find a `.vecs` file containing the embeddings and a `.metadata` file containing the tags associated with the vectors. These files then can be laoded to http://projector.tensorflow.org/ . Moreover you need to specify the names of the tags via `fileds` argument. E.g. if the documents were loaded from the ES newspapers they will be 'index', 'url'. 
Finally you can specify to create files for just a specific number of documents. Since most of the time documents are stored in a sorted order if you set `-n` documents will be randomly picked for nicer visualizations.


## TASKS

Since all tasks are python scripts simply running the main utilities, they will be called as python modules. Thus you can use them this way:

    $ python -m tasks.<script name> 

### Newspaper detection

This is for the task of predicting from the document embeddings to newspaper who wrote the document. Pleae use

    $ python -m tasks.accuracy_newspapers.py -c ./configs/test_newspapers.config

This will load the defined test set. Then vectors for the test set are inferred (10) a majority vote is taken to make the predictions. Then the `TPR` (# correct preditions / length of test set) per newspaper of each model is printed.


## REFERENCES

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.

Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents. http://arxiv.org/pdf/1405.4053v2.pdf
