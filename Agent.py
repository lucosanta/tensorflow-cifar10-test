
import argparse
import os
import re
import sys
import tarfile
import tensorflow as tf

from six.moves import urllib
import pickle

from Enumerators import Mode

from CNN import CNN
from file_loader import File_loader

class Agent(object):



    def __init__(self,log, mode = Mode.TRAIN):

        self.logger = log

        self.logger.debug('[Agent][Constructor] '+mode.name)

        self.model = None



    def train(self,args,keep_prob= 0.5,epochs= 1000,image_shape = (32,32,3)):
        self.logger.info('[Agent][train]')
        self.logger.info(args)

        # Remove previous weights, bias, inputs, etc..
        #tf.reset_default_graph()
        queue_loader = File_loader(batch_size=args.b_size, num_epochs=args.ep)

        model = CNN(args.lr, args.b_size, queue_loader.num_batches)
        model.build(queue_loader.images)
        model.loss(queue_loader.labels)
        train_op = model.train()
        model.accuracy(queue_loader.labels, model.logits)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        self.logger.info('Start training')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
        test_writer = tf.summary.FileWriter('./summary/test')

        try:
            ep = 0
            step = 1
            while not coord.should_stop():
                #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #run_metadata = tf.RunMetadata()

                summary, loss, accuracy, _ = sess.run([merged, model.loss_op, model.accuracy_op, train_op])

                print('epoch: %2d, step: %2d, loss: %.4f accuracy: %.4f' % (ep + 1, step, loss, accuracy))

                if step % queue_loader.num_batches == 0:
                    print('epoch: %2d, step: %2d, loss: %.4f, epoch %2d done.' % (ep + 1, step, loss, ep + 1))
                    checkpoint_path = os.path.join('./summary/log', 'cifar.ckpt')
                    saver.save(sess, checkpoint_path, global_step=ep + 1)
                    step = 1
                    train_writer.add_summary(summary, ep)
                    ep += 1
                else:
                    step += 1
        except tf.errors.OutOfRangeError:
            print('\nDone training, epoch limit: %d reached.' % (args.ep))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()
        print('Done')


    def eval(self):
        self.logger.info('[Agent][eval]')




    ######################################################################################

    def get_divided_datasets(self):
        '''
        I will take 4 data_batch file as input and 1 for validation. test_set is already provided

        :return: file names expected to exist in the input_dir
        '''
        file_names = {}
        file_names['train'] = ['data_batch_%d' % i for i in range(1, 5)]
        file_names['validation'] = ['data_batch_5']
        file_names['eval'] = ['test_batch']
        return file_names


    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def train_neural_network(self,session, optimizer, keep_probability, feature_batch, label_batch):
        """
        Optimize the session on a batch of images and labels
        : session: Current TensorFlow session
        : optimizer: TensorFlow optimizer function
        : keep_probability: keep probability
        : feature_batch: Batch of Numpy image data
        : label_batch: Batch of Numpy label data
        """
        print(feature_batch)
        print(label_batch)



    def print_stats(self,session, feature_batch, label_batch, cost, accuracy):
        """
        Print information about loss and validation accuracy
        : session: Current TensorFlow session
        : feature_batch: Batch of Numpy image data
        : label_batch: Batch of Numpy label data
        : cost: TensorFlow cost function
        : accuracy: TensorFlow accuracy function
        """
        loss = session.run(cost, feed_dict={
            x: feature_batch,
            y: label_batch,
            keep_prob: 1.})
        valid_acc = session.run(accuracy, feed_dict={
            x: valid_features,
            y: valid_labels,
            keep_prob: 1.})

        print('Loss: {:>10.4f} | Validation Accuracy: {:.4f}'.format(
            loss,
            valid_acc))