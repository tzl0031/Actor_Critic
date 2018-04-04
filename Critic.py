import numpy as np
import tensorflow as tf

GAMMA = 0.9


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        """
        Critic graph. State is input and value of state is output.
        :param sess:
        :param n_features:
        :param lr:
        """
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.placeholder(tf.float32, [1, 1], "v_next"))
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope("Critic"):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name="V"
            )

        # v
        with tf.variable_scope("squared_TD_error"):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)

        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        """
        learn the value of state
        evaluate by TD_error
        :param s:
        :param r:
        :param s_:
        :return:
        """
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_:v_, self.r: r})

        return td_error
