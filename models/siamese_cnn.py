import tensorflow as tf
import numpy as np
import os,sys

class siamesecnn(object):

    def __init__(self,seq_len,num_classes,pretrained_embeddings,filter_sizes,num_filters,l2_reg=0.0):

        self.input_x1=tf.placeholder(tf.int32,[None,seq_len],name='x1')
        self.input_x2=tf.placeholder(tf.int32,[None,seq_len],name='x2')
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name='labels')
        self.drop_prob=tf.placeholder(tf.float32,name='dropout')
        self.x1_len=tf.placeholder(tf.int32,[None],name='x1_len')
        self.x2_len=tf.placeholder(tf.int32,[None],name='x2_len')

        loss=tf.constant(0.0)

        embed_type=tf.Variable(pretrained_embeddings)
        embedding_dims=pretrained_embeddings.shape[1]


        lookup=tf.nn.embedding_lookup(embed_type,self.input_x1)
        embedding_x1=tf.expand_dims(lookup,-1)
        lookup=tf.nn.embedding_lookup(embed_type,self.input_x2)
        embedding_x2=tf.expand_dims(lookup,-1)

        r1=self.create_tower(embedding_x1,num_filters,filter_sizes,seq_len,embedding_dims,"0",False)
        r2=self.create_tower(embedding_x2,num_filters,filter_sizes,seq_len,embedding_dims,"0",True)

        l2_loss=tf.constant(0.0)

        with tf.name_scope('output'):
            feature_vec=tf.concat([r1,r2,tf.abs(r1-r2),tf.multiply(r1,r2)],1)
            feature_len=num_filters*len(filter_sizes)
            feature_len=4*feature_len

            num_hidden=int(np.sqrt(feature_len))

            W3=tf.get_variable('W3',shape=[feature_len,num_hidden],initializer=tf.contrib.layers.xavier_initializer())

            b3=tf.Variable(tf.constant(0.1,shape=[num_hidden]),name='b3')

            hid=tf.nn.relu(tf.nn.xw_plus_b(feature_vec,W3,b3))

            W4=tf.get_variable('W4',shape=[num_hidden,num_classes],initializer=tf.contrib.layers.xavier_initializer())

            b4=tf.Variable(tf.constant(0.1,shape=[num_classes]))

            l2_loss+=tf.nn.l2_loss(W4)
            l2_loss+=tf.nn.l2_loss(b4)

            self.scores=tf.nn.xw_plus_b(hid,W4,b4,name='scores')
            self.predictions=tf.argmax(self.scores,1,name='predictions')

        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits(self.scores,self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_reg*l2_loss

        with tf.name_scope('accuracy'):
            self.y_labels=tf.argmax(self.input_y,1,name='y_label')
            self.correct=tf.equal(self.predictions,self.y_labels,name='correct_predict')
            self.accuracy=tf.reduce_mean(tf.cast(self.correct,'float'),name='accuracy')

    def create_tower(self, embeddings, num_filters, filter_sizes, seq_len, embedding_dims, qid, reuse):
        W_name = "W" + qid
        b_name = "b" + qid
        h_name = "h" + qid
        conv_name = "conv" + qid
        pool_name = "pool" + qid

        with tf.variable_scope('inference', reuse=reuse):
            pooled_output = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-maxpool-%s-%s' % (qid, filter_size)):
                    filter_shape = [filter_size, embedding_dims, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name=W_name)
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name=b_name)
                    conv = tf.nn.conv2d(embeddings, W,strides=[1, 1, 1, 1], padding="VALID", name=conv_name)
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name=h_name)

                    max_pool = tf.nn.max_pool(h, [1, seq_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                              padding="VALID", name=pool_name)

                    pooled_output.append(max_pool)

            len_filters = num_filters * len(filter_sizes)
            h_pool = tf.concat(pooled_output,3)
            pool_flat = tf.reshape(h_pool, [-1, len_filters])

            with tf.name_scope("dropout-%s" % qid):
                h_drop = tf.nn.dropout(pool_flat, self.drop_prob)

        return h_drop


