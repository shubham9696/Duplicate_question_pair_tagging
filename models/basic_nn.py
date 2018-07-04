import tensorflow as tf
import numpy
import os,sys

class baseline_nn(object):
    def __init__(self,seq_len,num_classes,pretrained_embeddings):

        self.x1=tf.placeholder(tf.int32,[None,seq_len],name='x1')
        self.x2=tf.placeholder(tf.int32,[None,seq_len],name='x2')
        self.dropout_prob=tf.placeholder(tf.float32,[None],name='drop')
        self.y=tf.placeholder(tf.float32,[None,num_classes],name='labels')
        self.x1_length=tf.placeholder(tf.int32,[None],name='len1')
        self.x2_length=tf.placeholder(tf.int32,[None],name='len2')


        word_embeddings=tf.Variable(pretrained_embeddings)

        x1_embeddings=tf.nn.embedding_lookup(word_embeddings,self.x1)
        x2_embeddings=tf.nn.embedding_lookup(word_embeddings,self.x2)

        r1=tf.reduce_mean(x1_embeddings,axis=1)
        r2=tf.reduce_mean(x2_embeddings,axis=1)

        features=tf.concat([r1,r2,r-r2,tf.multiply(r1,r2)],1)

        output=tf.contrib.layers.fully_connected(features,num_classes,activation_fn=None)
        predict=tf.nn.softmax(output)

        with tf.name_scope('output'):
            self.scores=predict
            self.predictions=tf.argmax(predict,axis=1)

        with tf.name_scope('loss'):
            loss_vec=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.y)
            self.loss=tf.reduce_mean(loss_vec)

        with tf.name_scope('accuracy'):
            self.y_label=tf.argmax(self.y,1)
            correct_predict=tf.equal(self.predictions,self.y_label)
            self.accuracy=tf.reduce_mean(tf.cast(correct_predict,'float'),name='accuracy')


