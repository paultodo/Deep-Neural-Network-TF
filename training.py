# -*- coding: utf-8 -*-

### Training ###
import tensorflow as tf
import numpy as np
from model_nn import nn_model
from load_data import load_the_data
import datetime

from sklearn.metrics import confusion_matrix
from six.moves import cPickle as pickle
import math
import time
import os


def accuracy_numpy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# Model parameters
tf.flags.DEFINE_integer("nb_hidden_1", 1024, "Number of hidden nodes for layer 1 (default: 1024)")
tf.flags.DEFINE_integer("nb_hidden_2", 1024, "Number of hidden nodes for layer 2 (default: 800)")
tf.flags.DEFINE_integer("nb_hidden_3", 512, "Number of hidden nodes for layer 3 (default: 512)")
tf.flags.DEFINE_float("keep_prob_layer1", 0.9, "Probability to keep nodes in layer 1 (default: 0.8)")
tf.flags.DEFINE_float("keep_prob_layer2", 0.8, "Probability to keep nodes in layer 2 (default: 0.8)")
tf.flags.DEFINE_float("keep_prob_layer3", 0.7, "Probability to keep nodes in layer 3 (default: 0.6)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 256)")
tf.flags.DEFINE_integer("nb_steps", 301, "Number of training step (default: 1001)")
tf.flags.DEFINE_integer("eval_every", 100, "Number of steps between every eval print (default: 100)")
tf.flags.DEFINE_float("learning_rate", 0.05, "Initial learning rate (default: 0.0005)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.00000, "L2 regularizaion lambda (default: 0.0001)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


### LOAD DATA ###



train_dataset, train_labels, test_dataset, test_labels = load_the_data(which = 'mnist')
num_labels = test_labels.shape[1]


print ("...data loaded!")
# Compute var for init 
sqrt_0 = math.sqrt(2.0 / float(train_dataset.shape[1]))
sqrt_1 = math.sqrt(2.0 / float(FLAGS.nb_hidden_1))
sqrt_2 = math.sqrt(2.0 / float(FLAGS.nb_hidden_2))
sqrt_3 = math.sqrt(2.0 / float(FLAGS.nb_hidden_3))

print ("...Start training !")

num_steps = FLAGS.nb_steps

with tf.Graph().as_default():
    with tf.Session() as session:
        dnn = nn_model(
            dp1 = FLAGS.keep_prob_layer1,
            dp2 = FLAGS.keep_prob_layer2,
            dp3 = FLAGS.keep_prob_layer3,
            num_hidden1 = FLAGS.nb_hidden_1,
            num_hidden2 = FLAGS.nb_hidden_2,
            num_hidden3 = FLAGS.nb_hidden_3,
            data_set_size = train_dataset.shape[0],
            num_classes = num_labels,
            nb_features = train_dataset.shape[1],
            l2_reg_lambda = FLAGS.l2_reg_lambda,
            test_dataset = test_dataset,
            stdev_init = [sqrt_0, sqrt_1, sqrt_2, sqrt_3],
            batch_ = FLAGS.batch_size)

        global_step = tf.Variable(0)
        init_lr = FLAGS.learning_rate
        lr = tf.train.exponential_decay(init_lr, global_step, 500, 0.90, staircase=True)
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        grads_and_vars = optimizer.compute_gradients(dnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", dnn.loss)
        acc_summary = tf.scalar_summary("accuracy", dnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, session.graph_def)

        saver = tf.train.Saver(tf.all_variables())
        
        tf.initialize_all_variables().run()
        print("Initialized")

        neg_indices = np.where(train_labels == [0,1])[0]
        pos_indices = np.where(train_labels == [1,0])[0]
        batch_size = FLAGS.batch_size
        num_steps = FLAGS.nb_steps

        print "let's go for %d !" % num_steps
        t = time.time()
        for step in range(num_steps):


            batch_indices = np.random.choice(np.arange(train_dataset.shape[0]), size = batch_size)
            batch_data = train_dataset[batch_indices, :]
            batch_labels = train_labels[batch_indices, :]

            feed_dict = {dnn.input : batch_data, dnn.labels : batch_labels}
            _, l, accuracy_, summaries, predictions= session.run(
              [train_op, dnn.loss, dnn.accuracy, train_summary_op, dnn.train_prediction], feed_dict=feed_dict)
            train_summary_writer.add_summary(summaries, step)
            
            if (step % FLAGS.eval_every == 0):
                print "Last %i steps took" % FLAGS.eval_every, time.time() - t
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy_numpy(predictions, batch_labels))
                print("Test accuracy: %.1f%%" % accuracy_numpy(dnn.test_prediction.eval(), test_labels))
                print "Current learning rate", lr.eval()
                t = time.time()
                print "==========="
        
        print(confusion_matrix(np.argmax(dnn.test_prediction.eval(),1), np.argmax(test_labels,1)))
        
