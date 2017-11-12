
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import sys
import argparse

import numpy as np
import tensorflow as tf

from Agent import Agent

import logging
from Enumerators import DebugLevel,Mode
from LoggerConfigurator import configure_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=1000, help='number of epochs.')
    parser.add_argument('--batch-size', metavar='', type=int, default=200, help='batch size.')
    parser.add_argument('--dataset-url',metavar='',type=str, default='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', help='CIFAR URL')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: sys.exit('Unknown argument: {}'.format(unparsed))

    run = 1
    debug = DebugLevel.VERBOSE
    dataset_directory = './dataset'
    keep_prob = 0.8
    image_shape = (32, 32, 3)

    # Configuration of the logger
    logger = configure_log('lsantonastasi - cifar-10', debug=DebugLevel.VERBOSE)

    a = Agent(log=logger, mode=Mode.TRAIN)

    if args.train:
        a.train(args)
    if not args.train:
        parser.print_help()



