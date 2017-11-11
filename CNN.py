import numpy as np
import tensorflow as tf

class CNN:

    def __init__(self, lr=1e-5, batch_size=512, num_batches=97):
        self.lr = lr
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.g_step = tf.contrib.framework.get_or_create_global_step()
        self.train_var = True

    def build(self, x, train=True):
        '''
        net = self.conv_layer(images, 5, 3, 64, name='conv1')
        net = self.max_pool(net, name='pool1')
        net = self.lrn_layer(net, name='lrn1')

        net = self.conv_layer(net, 5, 64, 64, name='conv2')
        net = self.lrn_layer(net, name='lrn2')
        net = self.max_pool(net, name='pool2')

        dim = np.prod(net.shape[1:]).value
        net = tf.nn.relu(self.fc_layer(tf.reshape(net, [-1, dim]), dim, 384, name='fc3'))
        net = tf.nn.relu(self.fc_layer(net, 384, 192, name='fc4'))
        self.logits = self.fc_layer(net, 192, 10, name='logits')
        '''

        self.train_var = train
        # CNN layer 1 -- 32x32x3 to 16x16x48
        conv_layer = self.conv_layer(x, 2, 3, 48, name='conv1',strides=2)
        # CNN layer 2 -- 16x16x48 to 8x8x192
        conv_layer = self.conv_layer(conv_layer, 2, 48, 192, name='conv2',strides=2)#conv2d_maxpool(conv_layer, 192, (2, 2), conv_strides, pool_ksize, (1, 1))

        # CNN layer 3 -- 8x8x192 to 2x2x512
        conv_layer = self.conv_layer(conv_layer, 4, 192, 512, name='conv3',strides=4)#conv2d_maxpool(conv_layer, 384, (2, 2), conv_strides, (1, 1), pool_strides)


        # Function Definition from Above:
        #   flatten(x_tensor)

        # 2x2x512 to 2048
        conv_layer = self.flatten(conv_layer)

        #    Play around with different number of outputs
        # Function Definition from Above:
        #   fully_conn(x_tensor, num_outputs)

        # Dropout -- 3072 to (keep_prob * 3072)
        #fully_conn_layer = tf.nn.dropout(conv_layer, keep_prob)

        # Fully connected layer 1 -- (2048) to 384
        net = tf.nn.relu(self.fc_layer(conv_layer, 2048, 384, name='fc3'))

        # Fully connected layer 2 -- 384 to 192
        net = tf.nn.relu(self.fc_layer(net, 384, 192, name='fc4'))

        #Output
        logits = self.fc_layer(net, 192, 10, name='logits')

        return logits

    def loss(self, labels,logits,name=''):
        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss_op = tf.reduce_mean(cross_entropy_loss)
        loss_summary = tf.summary.scalar('loss', loss_op)
        return loss_op,loss_summary

    def accuracy(self, labels,logits,name=''):
        correct_prediction = tf.equal(tf.cast(labels, tf.int64), tf.argmax(logits, dimension=1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy_op)
        return accuracy_op, accuracy_summary

    def train(self,loss_op):
        self.lr = tf.train.exponential_decay(self.lr, self.g_step,
                self.batch_size * self.num_batches, 0.8, staircase=True)

        lr_summary=tf.summary.scalar('learning_rate', self.lr)
        return self.optimizer(self.lr).minimize(loss_op, global_step=self.g_step),lr_summary


    def optimizer(self, *args):
        return tf.train.AdamOptimizer(*args)

    def max_pool(self, bottom, name, k=3, s=2):
        return tf.nn.max_pool(bottom, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

    def lrn_layer(self, bottom, name, bias=1.0, alpha=1e-3/9.0, beta=0.75):
        return tf.nn.local_response_normalization(bottom, depth_radius=4, bias=bias, alpha=alpha, beta=beta, name=name)

    def conv_layer(self, bottom, f_size, in_c, out_c, name, strides=1):
        if self.train_var:
            with tf.variable_scope(name) as scope :
                f, b = self.get_conv_var(f_size, in_c, out_c, name)
                conv = tf.nn.conv2d(bottom, f, [1, strides, strides, 1], padding='SAME')
                return tf.nn.relu(tf.nn.bias_add(conv, b), name=scope.name)
        else:
            with tf.variable_scope(name,reuse=True) as scope :
                f, b = self.get_conv_var(f_size, in_c, out_c, name)
                conv = tf.nn.conv2d(bottom, f, [1, strides, strides, 1], padding='SAME')
                return tf.nn.relu(tf.nn.bias_add(conv, b), name=scope.name)

    def fc_layer(self, bottom, in_size, out_size, name):
        if self.train_var:
            with tf.variable_scope(name):
                w, b = self.get_fc_var(in_size, out_size, name)
                x = tf.reshape(bottom, [-1, in_size])
                return tf.nn.xw_plus_b(x, w, b)
        else:
            with tf.variable_scope(name, reuse=True) :
                w, b = self.get_fc_var(in_size, out_size, name)
                x = tf.reshape(bottom, [-1, in_size])
                return tf.nn.xw_plus_b(x, w, b)

    def get_conv_var(self, f_size, in_c, out_c, name):
        f = tf.get_variable(name+'_f', [f_size, f_size, in_c, out_c], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name+'_b', [out_c], initializer=tf.truncated_normal_initializer())
        return f, b

    def get_fc_var(self, in_size, out_size, name):
        w = tf.get_variable(name+'_w', [in_size, out_size], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name+'_b', [out_size], initializer=tf.truncated_normal_initializer())
        return w, b

    def flatten(self,x_tensor):
        '''
        Flatten x_tensor to (Batch Size, Flattened Image Size)
        : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
        : return: A tensor of size (Batch Size, Flattened Image Size).
        '''
        return tf.contrib.layers.flatten(x_tensor)



