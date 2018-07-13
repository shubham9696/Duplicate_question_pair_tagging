import numpy as np
import re
import itertools
import os.path
from collections import Counter
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


NUM = "<NUM>"


def clean_str(string, lower=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    result = string.strip()
    if lower:
        result = result.lower()
    return result


def load_data_and_labels(training_data_file):

    training_data=list(open(training_data_file,'r',encoding="utf8").readlines())
    training_data=[s.strip() for s in training_data]


    q1=[]
    q2=[]
    q1_lengths=[]
    q2_lengths=[]
    labels=[]

    for line in training_data:
        elements=line.split('\t')

        q1_length=len(elements[1].split())
        q2_lenght=len(elements[2].split())

        if q1_length>59 or q2_lenght>59:
            continue

        q1.append(elements[1].lower())
        q2.append(elements[2].lower())
        labels.append([0,1] if elements[0]==1 else [1,0])
        q1_lengths.append(q1_length)
        q2_lengths.append(q2_lenght)

    labels=np.concatenate([labels],0)
    q1_lengths=np.concatenate([q1_lengths],0)
    q2_lengths=np.concatenate([q2_lengths],0)

    return [q1,q2,labels,q1_lengths,q2_lengths]

def batch_iter(data,batch_size,shuffle=True):

    data=np.array(data)
    data_size=len(data)
    num_batches=int((data_size-1)/batch_size)+1

    if shuffle:
        shuffle_index=np.random.permutation(np.arange(data_size))
        shuffled_data=data[shuffle_index]
    else:
        shuffled_data=data

    for batch_num in range(num_batches):
        start=batch_num*batch_size
        end=min((batch_num+1)*batch_size,data_size)
        yield data[start:end]

def load_word_vector_mapping(embedding_file):

    ret=OrderedDict()
    for row in list(open(embedding_file,'r',encoding="utf8").readlines()):
        elements=row.strip().split()
        vocab=elements[0]
        ret[vocab]=np.array(list(map(float,elements[1:])))
    return ret

def normalize(word):
    return word.lower()

def load_embeddings(embedding_file,vocab_dict,embedding_dim,use_cache=True):
    embedding_cache_file=embedding_file+".cache.npy"
    if use_cache and os.path.isfile(embedding_cache_file):
        embeddings=np.load(embedding_cache_file)
        return embeddings

    embeddings=np.array(np.random.rand(len(vocab_dict)+1,embedding_dim),dtype=np.float32)
    embeddings[0]=0.
    for word,vec in load_word_vector_mapping(embedding_file).items():
        word=normalize(word)
        if word in vocab_dict:
            embeddings[vocab_dict[word]]=vec


    np.save(embedding_cache_file,embeddings)
    logger.info("Initialized embeddings.")
    return embeddings





