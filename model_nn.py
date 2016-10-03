# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class nn_model(object):
    """
    A deep NN for image classif
    Try to achieve best performance on MNIST without using convolution
    """
    def __init__(self, stdev_init, num_classes, batch_, test_dataset, data_set_size, nb_features, num_hidden1, num_hidden2, num_hidden3, dp1, dp2, dp3, l2_reg_lambda):

        self.input = tf.placeholder(tf.float32, shape=(batch_, nb_features))
        self.labels = tf.placeholder(tf.float32, shape=(batch_, num_classes))
        self.test_dataset = tf.constant(test_dataset)
        
        # Variables.
        w1 = tf.Variable(tf.truncated_normal([nb_features, num_hidden1], stddev = stdev_init[0]))
        b1 = tf.Variable(tf.zeros([num_hidden1]))
      
        w2 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev = stdev_init[1]))
        b2 = tf.Variable(tf.zeros([num_hidden2]))
        
        w3 = tf.Variable(tf.truncated_normal([num_hidden2, num_hidden3], stddev = stdev_init[2]))
        b3 = tf.Variable(tf.zeros([num_hidden3])) 
        
        w4 = tf.Variable(tf.truncated_normal([num_hidden3, num_classes], stddev = stdev_init[3]))
        b4 = tf.Variable(tf.zeros([num_classes]))
    
        def get_nn_model(dataset, use_dropout, batch_normalization, training = True) :
                
            if batch_normalization:
                h1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(dataset,w1) + b1, is_training = training, updates_collections = None))
            else :
                print("data", dataset.get_shape)
                print ("w1", w1.get_shape)
                h1 = tf.nn.relu(tf.matmul(dataset, w1) + b1)
            if use_dropout:
                logits_h1 = tf.nn.dropout(h1, dp1)
            else : 
                logits_h1 = h1
            
            if batch_normalization:
                h2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(logits_h1,w2) + b2, is_training = training, updates_collections = None))
            else:
                h2 = tf.nn.relu(tf.matmul(logits_h1, w2) + b2)
            if use_dropout:
                logits_h2 = tf.nn.dropout(h2, dp2)
            else :
                logits_h2 = h2
            
            if batch_normalization:
                h3 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(logits_h2, w3) + b3, is_training = training, updates_collections = None))
            else:
                h3 = tf.nn.relu(tf.matmul(logits_h2, w3) + b3)
            if use_dropout:
                logits_h3 = tf.nn.dropout(h3,dp3)
            else:
                logits_h3 = h3
            
            if batch_normalization:
                logits = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(logits_h3, w4) + b4, is_training = training, updates_collections = None))
            else:
                logits = tf.matmul(logits_h3, w4) + b4
            return logits


        def build_NN_3_hidden(x):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu, reuse = True):
                x = slim.fully_connected(x, 32, scope='fc/fc_1')
                x = slim.fully_connected(x, 64, scope='fc/fc_2')
                x = slim.fully_connected(x, 128, scope='fc/fc_3')
                x = slim.fully_connected(x, 10, scope='fc/fc_4')
                # slim.stack(x, slim.fully_connected, [32, 64, 128], scope='fc')
            return x

        # self.logits_train = get_nn_model(self.input, True, True, True)
        self.logits_train = get_nn_model(self.input, True, True)
        self.reg = l2_reg_lambda * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits_train, self.labels)) + self.reg
        

    
        # Predictions for the training, validation, and test data.
        self.train_prediction = tf.nn.softmax(self.logits_train)
        #valid_prediction = tf.nn.softmax(get_nn_model(tf_valid_dataset, False, True))
        self.test_prediction = tf.nn.softmax(get_nn_model(self.test_dataset, False, True))
        # self.test_prediction = 0
       
        self.predictions_label = tf.argmax(self.logits_train, 1, name="predictions")
        #Accuracy
        correct_predictions = tf.equal(self.predictions_label, tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


