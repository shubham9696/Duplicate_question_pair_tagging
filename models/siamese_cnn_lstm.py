import tensorflow as tf
import numpy as np
import os,sys


class siamese_cnnlstm(object):

    def __init__(self,seq_len,filter_sizes,num_filters,num_classes,pretrained_embeddings,l2_reg=0.0):

        self.input_x1=tf.placeholder(tf.int32,shape=[None,seq_len],name='x1')
        self.input_x2=tf.placeholder(tf.int32,shape=[None,seq_len],name='x2')
        self.input_y=tf.placeholder(tf.float32,shape=[None,num_classes])
        self.drop_prob=tf.placeholder(tf.float32,name='dropout')
        self.x1_len=tf.placeholder(tf.int32,[None],name='x1_len')
        self.x2_len=tf.placeholder(tf.int32,[None],name='x2_len')


        l2_loss=0.0

        embed_type=tf.Variable(pretrained_embeddings)
        embedding_dims=pretrained_embeddings.shape[1]

        embedding_x1=tf.nn.embedding_lookup(embed_type,self.input_x1)
        embedding_x2=tf.nn.embedding_lookup(embed_type,self.input_x2)

        r1=self.create_tower(embedding_x1,seq_len,embedding_dims,filter_sizes,num_filters,self.x1_len,"0",False)
        r2=self.create_tower(embedding_x2,seq_len,embedding_dims,filter_sizes,num_filters,self.x2_len,"0",True)

        with tf.name_scope('output'):
            feature=tf.concat([r1,r2,r1-r2,tf.multiply(r1,r2)],1)
            filter_len=num_filters*len(filter_sizes)
            feature_len=4*filter_len

            num_hidden=int(np.sqrt(feature_len))

            W3=tf.get_variable('W3',shape=[feature_len,num_hidden],initializer=tf.contrib.layers.xavier_initializer())
            b3=tf.Variable(tf.constant(0.1,shape=[num_hidden]),name='b3')
            h1=tf.nn.relu(tf.nn.xw_plus_b(feature,W3,b3),name='hidden')

            W4=tf.get_variable('W4',shape=[num_hidden,num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b4=tf.Variable(tf.constant(0.1,shape=[num_classes]),name='b4')

            self.scores=tf.nn.xw_plus_b(h1,W4,b4)
            self.predictions=tf.argmax(self.scores,1)

            l2_loss+=tf.nn.l2_loss(W4)
            l2_loss+=tf.nn.l2_loss(b4)

        with tf.name_scope('loss'):
            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_reg*l2_loss

        with tf.name_scope('accuracy'):
            self.y_labels = tf.arg_max(self.input_y, 1, name='y_labels')
            self.correct = tf.equal(self.predictions, self.y_labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))

    def create_tower(self,embedding,seq_len,embedding_dims,filter_sizes,num_filters,x_length,qid,reuse):
        W_name="W"+qid
        b_name="b"+qid
        h_name="h"+qid
        conv_name="conv"+qid
        pool_name="pool"+qid

        with tf.variable_scope('inference',reuse=reuse):
            with tf.name_scope('lstm'):
                x=embedding

                state_size=embedding_dims
                cell=tf.nn.rnn_cell.LSTMCell(num_units=state_size,state_is_tuple=True)

                output,state=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,inputs=x,sequence_length=x_length,dtype=tf.float32)

                output_fw,output_bw=output
                state_fw,state_bw=state

                encoding=tf.stack([output_fw,output_bw],axis=3)

                print("encoded: ",encoding.get_shape())

                pooled_outputs=[]

                for i,filter_size in enumerate(filter_sizes):
                    with tf.name_scope("conv-maxpool-%s-%s" %(qid, filter_size)):
                        filter_shape=[filter_size,embedding_dims,2,num_filters]
                        W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name=W_name)
                        b=tf.Variable(tf.constant(0.1,shape=[num_filters]),name=b_name)
                        conv=tf.nn.conv2d(encoding,W,strides=[1,1,1,1],padding='VALID',name=conv_name)

                        H=tf.nn.relu(tf.nn.bias_add(conv,b),name=h_name)

                        pooled=tf.nn.max_pool(H,ksize=[1,seq_len-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name=pool_name)

                        pooled_outputs.append(pooled)

                filter_len=num_filters*len(filter_sizes)
                h_pool=tf.concat(pooled_outputs,3)
                pool_flat=tf.reshape(h_pool,[-1,filter_len])

                with tf.name_scope('dropout'):
                    h_drop=tf.nn.dropout(pool_flat,self.drop_prob)

        return h_drop









