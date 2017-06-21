# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from six.moves import cPickle
import collections, os, sys, time

import tensorflow as tf

import reader
import model as md

def train(args):
    # Initialize the text reader
    loader = reader.TextReader(args)

    # Preprocess the data
    loader.preprocess()

    with open(os.path.join(args.data_path, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.data_path, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((loader.chars, loader.vocab), f)

    model = md.Model(args, loader.vocab_size)

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        
        
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))
            loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for b in range(loader.num_batches):
                start = time.time()
                x, y = loader.next_batch()
                feed = {model.input_data: x, model.label_data: y}
                for i in range(len(model.initial_state)):
                    feed[model.initial_state[0][0]] = state[i][0]
                    feed[model.initial_state[1][1]] = state[i][1]
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)

                # instrument for tensorboard
                summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                writer.add_summary(summ, e * loader.num_batches + b)

                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(e * loader.num_batches + b,
                              args.num_epochs * loader.num_batches,
                              e, train_loss, end - start))
                
                if (e * loader.num_batches + b) % args.snapshot == 0:
                    # save for the last result
                    checkpoint_path = os.path.join(args.data_path, 'model_iter{}.ckpt'.format(e * loader.num_batches + b))
                    saver.save(sess, checkpoint_path,
                               global_step=e * loader.num_batches + b)
                    print("Saved model to {}".format(checkpoint_path))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # The directory containing the data
    parser.add_argument(
        '--data_path',
        type=str,
        help='The directory where to find the data to train on.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default="./.logs/",
        help='The directory where to find the data to train on.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default="./.save/",
        help='The directory where to save the processed data.'
    )

    # The batch size for the input vectors
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='The batch size that should be used for the input text.'
    )
    parser.add_argument(
        '--seq_length',
        type=int,
        default=50,
        help='The batch size that should be used for the input text.'
    )

	# Argument regarding the model that should be used for the neurons
    parser.add_argument(
        '--model',
        type=str,
        default='rnn',
        help='rnn, blstm, lstm, gru'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='The number of layers to initialize.'
    )
    parser.add_argument(
        '--rnn_size',
        type=int,
        default=512,
        help='The number of neurons for the rnn.'
    )
    parser.add_argument(
        '--output_keep_prob',
        type=float,
        default=1.0,
        help='probability of keeping weights in the hidden layer'
    )
    parser.add_argument(
        '--input_keep_prob',
        type=float,
        default=1.0,
        help='probability of keeping weights in the input layer'
    )
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=5.,
        help='clip gradients at this value'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='number of epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.002,
        help='learning rate'
    )
    parser.add_argument(
        '--decay_rate',
        type=float,
        default=0.97,
        help='decay rate for rmsprop'
    )
    parser.add_argument(
        '--snapshot',
        type=int,
        default=1000,
        help='Same every snapshot iterations.'
    )

    args = parser.parse_args()

    # 
    train(args)

if __name__ == '__main__':
    main()
