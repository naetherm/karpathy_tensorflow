from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from six.moves import cPickle
from six import text_type
import collections, os, sys, time

import tensorflow as tf

import model as md

def sample(args):
    with open(os.path.join(args.data_path, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.data_path, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    vocab_size = len(vocab)
    model = md.Model(saved_args, vocab_size=vocab_size, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.data_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, chars, vocab, args.n, args.prime,
                               args.sample).encode('utf-8'))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default="./.save/",
        help='The directory where to save the processed data.'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        help='The directory where to find the data to train on.'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=500,
        help='The number of chars to sample.'
    )
    parser.add_argument(
        '--prime',
        type=text_type,
        default=u' ',
        help='The prime text.'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=1,
        help='0: Use max at each timestep, 1: sample at each timestep, 2: sample on spaces.'
    )

    args = parser.parse_args()

    sample(args)

if __name__ == '__main__':
    main()