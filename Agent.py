
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
        self.batch_size = 200
        self.model = None



    def train(self,args,keep_prob= 0.5,epochs= 1000,image_shape = (32,32,3), url = ''):

        self.logger.info('[Agent][train]')
        self.logger.info(args)


        self.batch_size = args.batch_size

        # Remove previous weights, bias, inputs, etc..
        #tf.reset_default_graph()
        data_loader = File_loader(batch_size=args.batch_size, num_epochs=args.ep , url = args.dataset_url)
        test_loader = File_loader(batch_size=args.batch_size, num_epochs=args.ep,train=False,url = args.dataset-url)

        model = CNN(args.lr, args.batch_size, data_loader.num_batches)
        model_eval  = CNN(args.lr, args.batch_size, test_loader.num_batches)
        
        model.build(data_loader.images)
        model_eval.build(test_loader.images)


        model.loss(data_loader.labels)
        train_op = model.train()
        model.accuracy(data_loader.labels, model.logits)
        model_eval.accuracy(test_loader.labels,model_eval.logits,name='test')

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        self.logger.info('Start training')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_summary = tf.summary.merge([model.accuracy_summary, model.loss_summary])
        test_summary = tf.summary.merge([model_eval.accuracy_summary])

        train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
        test_writer = tf.summary.FileWriter('./summary/test')

        try:
            ep = 0
            step = 1
            while not coord.should_stop():
                #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #run_metadata = tf.RunMetadata()

                summary, loss, accuracy, _ = sess.run([merged, model.loss_op, model.accuracy_op, train_op])

                self.logger.info('epoch: %2d, step: %2d, loss: %.4f accuracy: %.4f' % (ep + 1, step, loss, accuracy))

                if step % data_loader.num_batches == 0:
                    self.logger.info('epoch: %2d, step: %2d, loss: %.4f, epoch %2d done.' % (ep + 1, step, loss, ep + 1))
                    checkpoint_path = os.path.join('./summary/log', 'cifar.ckpt')
                    saver.save(sess, checkpoint_path, global_step=ep + 1)
                    step = 1
                    train_writer.add_summary(summary, ep)
                    ep += 1
                else:
                    step += 1
        except tf.errors.OutOfRangeError:
            self.logger.info('\nDone training, epoch limit: %d reached.' % (args.ep))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()
        self.logger.info('Done')


    def evaluate_set (sess, top_k_op, num_examples):
        """Convenience function to run evaluation for for every batch. 
            Sum the number of correct predictions and output one precision value.
        Args:
            sess:          current Session
            top_k_op:      tensor of type tf.nn.in_top_k
            num_examples:  number of examples to evaluate
        """
        self.logger.info('[Agent][evaluate_set]')
        num_iter = int(math.ceil(num_examples / self.batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * self.batch_size

        for step in range(0,num_iter):
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)

        # Compute precision
        return true_count / total_sample_count


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