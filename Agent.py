
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
        test_loader = File_loader(batch_size=args.batch_size, num_epochs=args.ep,train=False,url = args.dataset_url)

        model = CNN(args.lr, args.batch_size, data_loader.num_batches)

        model.build(data_loader.images)
        model.loss(data_loader.labels)

        train_op = model.train()
        model.accuracy(data_loader.labels, model.logits)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        self.logger.info('Start training')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_summary = tf.summary.merge([model.accuracy_summary, model.loss_summary])

        train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
        test_writer = tf.summary.FileWriter('./summary/test')

        try:
            ep = 0
            step = 1
            while not coord.should_stop():
                #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #run_metadata = tf.RunMetadata()

                train_summary, loss, accuracy, _ = sess.run([train_summary, model.loss_op, model.accuracy_op, train_op])

                self.logger.info('epoch: %2d, step: %2d, loss: %.4f accuracy: %.4f' % (ep + 1, step, loss, accuracy))

                if step % data_loader.num_batches == 0:
                    self.logger.info('epoch: %2d, step: %2d, loss: %.4f, epoch %2d done.' % (ep + 1, step, loss, ep + 1))
                    checkpoint_path = os.path.join('./summary/log', 'cifar.ckpt')
                    saver.save(sess, checkpoint_path, global_step=ep + 1)
                    step = 1
                    train_writer.add_summary(train_summary, ep)
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


    def evaluate_model(self,sess, model, global_step, summary_writer, summary_op):
        """Computes perplexity-per-word over the evaluation dataset.
        Summaries and perplexity-per-word are written out to the eval directory.
        Args:
          sess: Session object.
          model: Instance of ShowAndTellModel; the model to evaluate.
          global_step: Integer; global step of the model checkpoint.
          summary_writer: Instance of FileWriter.
          summary_op: Op for generating model summaries.
        """
        # Log model summaries on a single batch.
        #summary_str = sess.run(summary_op)
        #summary_writer.add_summary(summary_str, global_step)

        # Compute perplexity over the entire dataset.
        num_eval_batches = int(math.ceil(10132 / self.batch_size))

        start_time = time.time()
        sum_losses = 0.
        sum_weights = 0.
        for i in range(0,num_eval_batches):
            accuracy, summary = sess.run([
                model.accuracy_op,
                model.accuracy_summary
            ])
            #sum_losses += np.sum(cross_entropy_losses * weights)
            #sum_weights += np.sum(weights)
            if not i % 100:
                tf.logging.info("Computed losses for %d of %d batches.", i + 1,
                                num_eval_batches)
        eval_time = time.time() - start_time

        perplexity = math.exp(sum_losses / sum_weights)
        tf.logging.info("Perplexity = %f (%.2g sec)", perplexity, eval_time)

        # Log perplexity to the FileWriter.
        summary = tf.Summary()
        value = summary.value.add()
        value.simple_value = perplexity
        value.tag = "Perplexity"
        summary_writer.add_summary(summary, global_step)

        # Write the Events file to the eval directory.
        #summary_writer.flush()
        tf.logging.info("Finished processing evaluation at global step %d.",
                        global_step)


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