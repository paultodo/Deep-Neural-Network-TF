import pandas as pd
import numpy as np

from IPython.display import display, HTML
import time; start_time = time.time()
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.externals import joblib
from scipy import sparse
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef

from six.moves import cPickle as pickle
import tflearn




def load_the_data(which = 'notmnist'):

	if which == 'mnist':
		import tflearn.datasets.mnist as mnist
		train_dataset, train_labels, test_dataset, test_labels = mnist.load_data(one_hot=True)
		print('Training set', train_dataset.shape, train_labels.shape)
		print('Test set', test_dataset.shape, test_labels.shape)
		return train_dataset, train_labels, test_dataset, test_labels

	elif which == 'kaggle':
		num_labels = 2
		print "Loading data..."
		feature_names = ['L3_S38_F3960', 'L3_S33_F3865', 'L3_S38_F3956', 'L3_S33_F3857',
	       'L3_S29_F3321', 'L1_S24_F1846', 'L3_S32_F3850', 'L3_S29_F3354',
	       'L3_S29_F3324', 'L3_S35_F3889', 'L0_S1_F28', 'L1_S24_F1844',
	       'L3_S29_F3376', 'L0_S0_F22', 'L3_S33_F3859', 'L3_S38_F3952', 
	       'L3_S30_F3754', 'L2_S26_F3113', 'L3_S30_F3759', 'L0_S5_F114']

		numeric_cols = pd.read_csv("../input/train_numeric.csv", nrows = 1).columns.values
		imp_idxs = [np.argwhere(feature_name == numeric_cols)[0][0] for feature_name in feature_names]
		train = pd.read_csv("../input/train_numeric.csv", 
		                index_col = 0, header = 0, usecols = [0, len(numeric_cols) - 1] + imp_idxs)

		numeric_cols_test = pd.read_csv("../input/test_numeric.csv", nrows = 1).columns.values
		imp_idxs = [np.argwhere(feature_name == numeric_cols_test)[0][0] for feature_name in feature_names]
		test = pd.read_csv("../input/test_numeric.csv", 
		                index_col = 0, header = 0, usecols = [0, len(numeric_cols_test) - 1] + imp_idxs)

		print "Using imputer"
		imputer = Imputer()

		total_train_dataset = train[feature_names].values[:50000].astype(np.float32)
		raw_total_train_labels = train['Response'].values[:50000].astype(np.float32)
		test_kaggle_dataset = test[feature_names].values[:50000].astype(np.float32)
		print ("...Data loaded !")
		print ("...Split train / test")

		print "split train test"
		sss = StratifiedShuffleSplit(raw_total_train_labels, 2, test_size=0.2, random_state=0)
		for train_index, test_index in sss:
		    train_dataset, test_dataset = total_train_dataset[train_index], total_train_dataset[test_index]
		    raw_train_labels, raw_test_labels = raw_total_train_labels[train_index], raw_total_train_labels[test_index]

		def reformat(labels):
		  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
		  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
		  return labels

		train_labels = reformat(raw_train_labels)
		test_labels = reformat(raw_test_labels)


		print ("Train dataset shape %i %i" % (train_dataset.shape[0], train_dataset.shape[1]))
		print ("Test dataset shape %i %i" % (test_dataset.shape[0], test_dataset.shape[1]))
		print ("Train labels shape %i %i" % (train_labels.shape[0], train_labels.shape[1]))
		print ("Test labels shape %i %i" % (test_labels.shape[0], test_labels.shape[1]))
		return train_dataset, train_labels, test_dataset, test_labels


	elif which == 'notmnist':
		num_labels = 10
		print ("loading data...")

		pickle_file = 'notMNIST.pickle'

		with open(pickle_file, 'rb') as f:
		  save = pickle.load(f)
		  train_dataset = save['train_dataset']
		  train_labels = save['train_labels']
		  valid_dataset = save['valid_dataset']
		  valid_labels = save['valid_labels']
		  test_dataset = save['test_dataset']
		  test_labels = save['test_labels']
		  del save  # hint to help gc free up memory
		  print('Training set', train_dataset.shape, train_labels.shape)
		  print('Validation set', valid_dataset.shape, valid_labels.shape)
		  print('Test set', test_dataset.shape, test_labels.shape)


		def reformat(dataset, labels):
		  dataset = dataset.reshape((-1, 784)).astype(np.float32)
		  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
		  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
		  return dataset, labels
		train_dataset, train_labels = reformat(train_dataset, train_labels)
		valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
		test_dataset, test_labels = reformat(test_dataset, test_labels)
		print('Training set', train_dataset.shape, train_labels.shape)
		print('Validation set', valid_dataset.shape, valid_labels.shape)
		print('Test set', test_dataset.shape, test_labels.shape)
		return train_dataset, train_labels, test_dataset, test_labels
