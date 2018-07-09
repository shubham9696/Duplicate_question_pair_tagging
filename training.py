import tensorflow as tf
import numpy as np
import os
import time
import sklearn as sk
import datetime
import data_helper
from tensorflow.contrib import learn

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from model.basic_nn import baseline_nn
from model.siamese_cnn import siamesecnn
from model.siamese_lstm import siameselstm
from model.siamese_cnn_lstm import siamese_cnnlstm

tf.flags.DEFINE_float("val_sample_percentage",0.1,"percentage of Validation data")
tf.flags.DEFINE_string("training_data_file","./data/training.full.tsv","Data for training")
tf.flags.DEFINE_string("embeddings_file","./glove.6B/glove.6B.100d.txt","Data for pretrained embeddings")

tf.flags.DEFINE_integer("embedding_dim",100,"word embedding dimension")
tf.flags.DEFINE_string("filter_sizes","3,4,5","comma separated filters sizes")
tf.flags.DEFINE_integer("num_filters",128,"number of filters per filter size")
tf.flags.DEFINE_float("dropout_prob",0.5,"Dropout probability")
tf.flags.DEFINE_float("l2_reg_lambda",0.3,"regularization constant")

tf.flags.DEFINE_integer("batch_size",256,"size of each batch")
tf.flags.DEFINE_integer("num_epochs",100,"number of training epochs")
tf.flags.DEFINE_integer("evaluate_every",100,"evaluate on dev set")
tf.flags.DEFINE_integer("checkpoint_every",100,"save model")
tf.flags.DEFINE_integer("num_checkpoints",5,"number of checkpoints")

tf.flags.DEFINE_boolean("allow_soft_placement",True,"Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement",False,"Log placement of ops on devices")
tf.flags.DEFINE_boolean("use_cached_embeddings",True,"Cache embeddings locally on disk for repeated runs")

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters")
for attr,value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

print("Loading Data...")
q1,q2,y,x1_length,x2_length=data_helper.load_data_and_labels(FLAGS.training_data_file)

max_length=max(max([len(x.split(" ")) for x in q1]),max([len(x.split(" ")) for x in q2]))
vocab_processor=learn.preprocessing.VocabularyProcessor(max_length)
print("max question length:",max_length)

#converting to embedding matrix

x_text=q1+q2
vocab_ids=np.array(list(vocab_processor.fit_transform(x_text)))
x1=vocab_ids[:len(q1)]
x2=vocab_ids[len(q1):]


print("Loading Word embeddings")
vocab_dict=vocab_processor.vocabulary_._mapping
pretrained_embeddings=data_helper.load_embeddings(FLAGS.embeddings_file,vocab_dict,FLAGS.embedding_dim,FLAGS.use_cached_embeddings)

print("Shuffling Data:")
np.random.seed(10)
shuffled_index=np.random.permutation(np.arange(len(y)))
x1_shuffled=x1[shuffled_index]
x2_shuffled=x2[shuffled_index]
y_shuffled=y[shuffled_index]
q1_lenghts_shuffled=x1_length[shuffled_index]
q2_lenghts_shuffled=x2_length[shuffled_index]

print("Splitting Training/Validation data")
validation_index=-1*int(FLAGS.val_sample_percentage*float(len(y)))
x1_training,x1_validation=x1_shuffled[:validation_index],x1_shuffled[validation_index:]
x2_training,x2_validation=x2_shuffled[:validation_index],x2_shuffled[validation_index:]
y_train,y_val=y_shuffled[:validation_index],y_shuffled[validation_index:]
x1_lengths_train,x1_lenghts_val=q1_lenghts_shuffled[:validation_index],q1_lenghts_shuffled[validation_index:]
x2_lengths_train,x2_lenghts_val=q2_lenghts_shuffled[:validation_index],q2_lenghts_shuffled[validation_index:]

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Validation Split: {:d}/{:d}".format(len(y_train),len(y_val)))

with tf.Graph().as_default():
    session_conf=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)
    sess=tf.Session(config=session_conf)

    with sess.as_default():
        #cnn=baseline_nn(seq_len=x1_training.shape[1],num_classes=y_train.shape[1],pretrained_embeddings=pretrained_embeddings)
        cnn = siamese_cnnlstm(
              seq_len=x1_training.shape[1],
              num_classes=y_train.shape[1],
              pretrained_embeddings=pretrained_embeddings,
              filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
              num_filters=FLAGS.num_filters,
              l2_reg=FLAGS.l2_reg_lambda)
        global_step=tf.Variable(0,name="global_step",trainable=False)

        optimizer=tf.train.AdamOptimizer(1e-3)
        grads_and_vars=optimizer.compute_gradients(cnn.loss)
        train=optimizer.apply_gradients(grads_and_vars,global_step=global_step)

        timestamp=str(int(time.time()))
        out_dir=os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary=tf.summary.scalar("loss",cnn.loss)
        accuracy_summary=tf.summary.scalar("accuracy",cnn.accuracy)

        train_summary=tf.summary.merge([loss_summary,accuracy_summary])
        train_summary_dir=os.path.join(out_dir,"summaries","train")
        train_summary_write=tf.summary.FileWriter(train_summary_dir,sess.graph)

        validation_summary=tf.summary.merge(([loss_summary,accuracy_summary]))
        validation_summary_dir=os.path.join(out_dir,"summaries","validation")
        validation_summary_write=tf.summary.FileWriter(validation_summary_dir,sess.graph)

        checkpoint_dir=os.path.abspath(os.path.join(out_dir,"checkpoint"))
        checkpoint_prefix=os.path.join(checkpoint_dir,"model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpointer=tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)

        vocab_processor.save(os.path.join(out_dir,"vocab"))

        sess.run(tf.global_variables_initializer())

        def train_step(x1_batch,x2_batch,y_batch,x1_lenghts_batch,x2_lengths_batch,epoch):

            feed_dict={
                cnn.input_x1:x1_batch,
                cnn.input_x2:x2_batch,
                cnn.input_y:y_batch,
                cnn.drop_prob:FLAGS.dropout_prob,
                cnn.x1_len:x1_lenghts_batch,
                cnn.x2_len:x2_lengths_batch,
            }

            _,step,summary,loss,accuracy,score,predictions,y_label=sess.run(
                [train,global_step,train_summary,cnn.loss,cnn.accuracy,cnn.scores,cnn.predictions,cnn.y_labels],
                feed_dict=feed_dict)

            labels=[0, 1]
            precision=precision_score(y_true=y_label,y_pred=predictions,labels=labels)
            recall=recall_score(y_true=y_label,y_pred=predictions,labels=labels)
            f1=f1_score(y_true=y_label,y_pred=predictions,labels=labels)
            time_str = datetime.datetime.now().isoformat()
            print("{}: \tepoch {}, \tstep {}, \tloss {:g}, \tacc {:g}, \tprec {:g}, \trec {:g}, \tf1 {:g}".format(time_str, epoch, step, loss, accuracy, precision, recall, f1))

            train_summary_write.add_summary(summary,step)


        def validation_step(x1_batch,x2_batch,y_batch,x1_lengths_batch,x2_lengths_batch,epoch,writer=None):

            feed_dict={
                cnn.input_x1:x1_batch,
                cnn.input_x2:x2_batch,
                cnn.input_y:y_batch,
                cnn.drop_prob:1.0,
                cnn.x1_len:x1_lengths_batch,
                cnn.x2_len:x2_lengths_batch,
            }

            step,summary,loss,accuracy,score,predictions,y_label=sess.run(
                [global_step,validation_summary,cnn.loss,cnn.accuracy,cnn.scores,cnn.predictions,cnn.y_labels],
                feed_dict=feed_dict)

            labels=[0,1]

            precision=precision_score(y_true=y_label,y_pred=predictions,labels=labels)
            recall=recall_score(y_true=y_label,y_pred=predictions,labels=labels)
            f1=f1_score(y_true=y_label,y_pred=predictions,labels=labels)

            time_str = datetime.datetime.now().isoformat()
            print("{}: \tepoch {}, \tstep {}, \tloss {:g}, \tacc {:g}, \tprec {:g}, \trec {:g}, \tf1 {:g}".format(time_str, epoch, step, loss, accuracy, precision, recall, f1))
            if writer:
                writer.add_summary(summary, step)

        dataset=list(zip(x1_training,x2_training,y_train,x1_lengths_train,x2_lengths_train))
        for epoch in range(FLAGS.num_epochs):
            print("**** Epoch: ",epoch)

            batches=data_helper.batch_iter(dataset,FLAGS.batch_size)

            for batch in batches:
                x1_batch,x2_batch,y_batch,x1_length_batch,x2_length_batch=zip(*batch)
                train_step(x1_batch,x2_batch,y_batch,x1_length_batch,x2_length_batch,epoch)
                current_step=tf.train.global_step(sess,global_step)
                #if current_step%FLAGS.evaluate_every==0:
                #    print("\nEvaluation:")
                #    validation_step(x1_validation,x2_validation,y_val,x1_lenghts_val,x2_lenghts_val,epoch,writer=validation_summary_write)
                #    print("")
                if current_step%FLAGS.checkpoint_every==0:
                    path=checkpointer.save(sess,checkpoint_prefix,global_step=current_step)
                    print("Saved model!!!")