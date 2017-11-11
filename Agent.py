
import os
import tensorflow as tf

import math
import numpy as np
import time

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
        test_loader = File_loader(batch_size=args.batch_size, num_epochs=args.ep, train=False,url=args.dataset_url)
        model = CNN(args.lr, args.batch_size, data_loader.num_batches)

        logits_tr = model.build(data_loader.images)
        logits_ev = model.build(test_loader.images,train=False)

        loss_tr, loss_summary_tr = model.loss(data_loader.labels,logits_tr)
        loss_ev, loss_summary_ev = model.loss(test_loader.labels,logits_ev,name='test')

        train_op,lr_summary = model.train(loss_tr)
        #   test_op = model.test(test_loader.images,test_loader.labels)
        accuracy_tr,accuracy_summary_tr = model.accuracy(data_loader.labels,logits_tr)
        accuracy_ev,accuracy_summary_ev = model.accuracy(test_loader.labels,logits_ev,name='test')

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        self.logger.info('Start training')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_tr = tf.summary.merge([accuracy_summary_tr, loss_summary_tr, data_loader.image_summary,lr_summary])
        summary_ev = tf.summary.merge([accuracy_summary_ev, loss_summary_ev, test_loader.image_summary  ])

        train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
        test_writer = tf.summary.FileWriter('./summary/test')
        try:
            ep = 0
            step = 1
            while not coord.should_stop():
                #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #run_metadata = tf.RunMetadata()

                train_summ, loss, accuracy, _ = sess.run(
                                    [summary_tr, loss_tr, accuracy_tr,train_op]
                )

                self.logger.info('epoch: %2d, step: %2d, loss: %.4f accuracy: %.4f' % (ep + 1, step, loss, accuracy))

                if step % data_loader.num_batches == 0:

                    checkpoint_path = os.path.join('./summary/log', 'cifar.ckpt')
                    saver.save(sess, checkpoint_path, global_step=ep + 1)
                    step = 1
                    train_writer.add_summary(train_summ, ep)

                    test_summ, loss_eva, accuracy_eva = sess.run(
                        [summary_ev, loss_ev, accuracy_ev]
                    )

                    test_writer.add_summary(test_summ,ep)
                    self.logger.info('-------------- Episode '+str(ep+1)+'--------------')
                    self.logger.info(
                        'loss: %.2f,loss test: %.2f,accuracy: %.2f,accuracy test: %.2f' % ( loss, loss_eva,accuracy,accuracy_eva))
                    self.logger.info('--------------------------------------------------')
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