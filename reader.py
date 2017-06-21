# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections, os, sys

import numpy as np
from six.moves import cPickle
import tensorflow as tf

class TextReader(object):

    def __init__(self, args):
        # Make a copy of the provided arguments
        self.args = args

        # Copy the data path
        self.data_path = args.data_path
        self.train_file = os.path.join(self.data_path, 'train.txt')
        # Save the data
        self.vocab_file = os.path.join(self.data_path, 'vocab.pkl')
        self.tensor_file = os.path.join(self.data_path, 'data.npy')

        # The batch size
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length

        self.raw_data = None
        self.chars = None 
        self.vocab_size = 0
        self.vocab = None
        self.tensor = None

        self.num_batches = 0
        self.x_batches = []
        self.y_batches = []
        self.batch_pointer = 0

    def load_preprocessed(self):
        with open(self.vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(self.tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

    def preprocess(self):
        '''
        Responsible for preprocessing the training data. Thereby this method is 
        responsible for reading the data and preprocessing it.
        '''

        # If the data was process once, just load it instead of processing it
        # once again
        if os.path.exists(self.vocab_file) and os.path.exists(self.tensor_file):
            self.load_preprocessed()
        else:
            # Read the training data from the file 'train.txt'
            with codecs.open(self.train_file, 'r', 'utf-8') as f:
                self.raw_data = f.read() #.encode('utf-8').strip()

            counter = collections.Counter(self.raw_data)
            count_pairs = sorted(counter.items(), key=lambda x: -x[1])
            self.chars, _ = zip(*count_pairs)
            self.vocab_size = len(self.chars)
            self.vocab = dict(zip(self.chars, range(len(self.chars))))
            self.tensor = np.array(list(map(self.vocab.get, self.raw_data)))

            # save the data
            with open(self.vocab_file, 'wb') as f:
                cPickle.dump(self.chars, f)
            np.save(self.tensor_file, self.tensor)

        # Create the batches
        self.create_batches()
        # To be sure: Reset the batch pointer
        self.reset_batch_pointer()

        print("Some information:")
        print("VocabSize:"+str(self.vocab_size))
        
    def create_batches(self):
        '''
        '''
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        np.set_printoptions(threshold=np.nan)
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        '''
        Returns the next batch
        '''
        x, y = self.x_batches[self.batch_pointer], self.y_batches[self.batch_pointer]

        # Increase the batch pointer counter
        self.batch_pointer += 1

        return x, y

    def reset_batch_pointer(self):
        '''
        Resets the batch pointer to zero.
        '''
        self.batch_pointer = 0
