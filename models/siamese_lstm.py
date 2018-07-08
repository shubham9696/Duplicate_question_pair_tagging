import tensorflow as tf
import numpy as np
import os,sys

class siameselstm(object):

    def __init__(self,num_classes,seq_len,num_filters,pretrained_embeddings,filter_sizes,l2_reg=0.0):

        self.input_x1=tf.placeholder(tf.int32,[None,seq_len],name='x1')
        self.input_x2=tf.placeholder(tf.int32,[None,seq_len],name='x2')
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name='labels')
        self.drop_prob=tf.placeholder(tf.float32,name='dropout')
        self.x1_len=tf.placeholder(tf.int32,[None])
        self.x2_len=tf.placeholder(tf.int32,[None])


        l2_loss=tf.constant(0.0)

        embed_type=tf.Variable(pretrained_embeddings)
        embed_dims=pretrained_embeddings.shape[1]

        embedding_x1=tf.nn.embedding_lookup(embed_type,self.input_x1)
        embedding_x2=tf.nn.embedding_lookup(embed_type,self.input_x2)

        r1=self.create_tower(embedding_x1,seq_len,filter_sizes,num_filters,self.x1_len,embed_dims,"0",False)
        r2=self.create_tower(embedding_x2,seq_len,filter_sizes,num_filters,self.x2_len,embed_dims,"0",True)

        with tf.name_scope("output"):

            features=tf.concat([r1,r2,r1-r2,tf.multiply(r1,r2)],1)
            feature_len=4*r1.get_shape().as_list()[1]

            num_hidden1=256   #multiple of 16
            num_hidden2=256

            W3=tf.get_variable('W3',shape=[feature_len,num_hidden1],initializer=tf.contrib.layers.xavier_initializer())
            b3=tf.Variable(tf.constant(0.1,shape=[num_hidden1]),name='b3')
            h3=tf.nn.relu(tf.nn.xw_plus_b(features,W3,b3),name='hidden1')

            W4=tf.get_variable('W4',shape=[num_hidden1,num_hidden2],initializer=tf.contrib.layers.xavier_initializer())
            b4=tf.Variable(tf.constant(0.1,shape=[num_hidden2]),name='b4')
            h4=tf.nn.relu(tf.nn.xw_plus_b(h3,W4,b4),name='hidden2')

            W5=tf.get_variable('W5',shape=[num_hidden2,num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b5=tf.Variable(tf.constant(0.1,shape=[num_classes]),name='b5')

            l2_loss+=tf.nn.l2_loss(W5)
            l2_loss+=tf.nn.l2_loss(b5)

            self.scores=tf.nn.xw_plus_b(h4,W5,b5,name='scores')
            self.predictions=tf.arg_max(self.scores,1,name='predictions')

        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits(self.scores,self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_reg*l2_loss

        with tf.name_scope('accuracy'):
            self.y_labels=tf.arg_max(self.input_y,1,name='y_labels')
            self.correct=tf.equal(self.predictions,self.y_labels,name='correct_predict')
            self.accuracy=tf.reduce_mean(tf.cast(self.correct,'float'))


    def create_tower(self,embeddings,seq_len,filter_sizes,num_filters,x_len,embedding_dims,qid,reuse):

        with tf.variable_scope("inference",reuse=reuse):
            with tf.name_scope('lstm'):
                x=embeddings

                state_size=5
                num_layers=3

                #cell=tf.nn.rnn_cell.LSTMCell(num_units=state_size,state_is_tuple=True)
                def lstm_cell():
                    lstm = tf.nn.rnn_cell.LSTMCell(num_units=state_size,state_is_tuple=True)
                    return lstm
                stacked_cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)],state_is_tuple=True)

                outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_cell,cell_bw=stacked_cell,dtype=tf.float32,inputs=x,sequence_length=x_len)

                output_fw,output_bw=outputs
                states_fw,states_bw=states

                encoding=tf.stack([output_fw,output_bw],axis=3)
                encoding=tf.concat(encoding,axis=3)
                encoding=tf.reshape(encoding,[-1,state_size*seq_len*2])

            with tf.name_scope('dropout'):
                output_drop=tf.nn.dropout(encoding,self.drop_prob)

        return output_drop



