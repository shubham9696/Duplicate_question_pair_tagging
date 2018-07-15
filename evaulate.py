import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from tensorflow.contrib import learn
import csv

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

tf.flags.DEFINE_string("testing_file","./data/test.full.tsv","data for evaluating")
tf.flags.DEFINE_string("embedding_file","./glove.6B/glove.6B.100d.txt","pretrained embediing file")

tf.flags.DEFINE_integer("batch_size",512,"size of batch")
tf.flags.DEFINE_string("checkpoint_dir","","checkpoint file")

tf.flags.DEFINE_boolean("allow_soft_placement",True,"Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement",False,"Log placement of ops on devices")

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

print("Loading Data...")
q1,q2,y_label,q1_length,q2_length=data_helper.load_data_and_labels(FLAGS.testing_file)

x_text=q1+q2
vocab_path=os.path.join(FLAGS.checkpoint_dir,"..","vocab")
vocab_processor=learn.preprocessing.VocabularyProcessor.restore(vocab_path)
vocab_ids=np.array(list(vocab_processor.tranform(x_text)))

x1_test=vocab_ids[:len(q1)]
x2_test=vocab_ids[len(q1):]
y_test=np.argmax(y_label,axis=1)

checkpoint_file=tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph=tf.Graph()
with graph.as_default():
    session_conf=tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess=tf.Session(config=session_conf)

    with sess.as_default():
        checkpointer=tf.train.import_meta_graph("{}meta".format(checkpoint_file))
        checkpointer.restore(sess,checkpoint_file)

        input_x1=graph.get_operation_by_name("input_x1").outputs[0]
        input_x2=graph.get_operation_by_name("input_x2").outputs[0]

        dropout_prob=graph.get_operation_by_name("drop_prob").outputs[0]
        input_x1_len=graph.get_operation_by_name("x1_len").outputs[0]
        input_x2_len=graph.get_operation_by_name("x2_len").outputs[0]

        predictions=graph.get_operation_by_name("output/predictions").outputs[0]

        batches=data_helper.batch_iter(list(zip(x1_test,x2_test,q1_length,q2_length)))

        all_predictions=[]

        for batch in batches:
            x1_batch,x2_batch,x1_length_batch,x2_length_batch=zip(*batch)
            batch_predictions=sess.run(predictions,
                                       feed_dict={
                                           input_x1:x1_batch,
                                           input_x2:x2_batch,
                                           dropout_prob:1.0,
                                           input_x1_len:x1_length_batch,
                                           input_x2_len:x2_length_batch
                                       })
            all_predictions.append(batch_predictions)

if y_test is not None:
    correct_prediction=float(sum(all_predictions==y_test))

    labels=[0,1]
    precision=precision_score(y_test,all_predictions,labels=labels)
    recall=recall_score(y_test,all_predictions,labels=labels)
    f1=f1_score(y_test,all_predictions,labels=labels)

    print("Total number of test examples: {}",len(y_test))
    accuracy=float(correct_prediction/len(y_test))
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1: {:.4f}".format(f1))

predictions_readable = np.column_stack((np.array(q1), np.array(q2), [int(x) for x in all_predictions], np.array(y_test)))
out_path=os.path.join(FLAGS.checkpoint_dir,"..","prediction.csv")
print("Saving")
with open(out_path,"w") as f:
    csv.writer(f,delimiter="\t").writerows(predictions_readable)





