
import numpy as np
import tensorflow as tf


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        """
        build Actor graph. State is input and probability of each action is output.
        :param sess:
        :param n_features:
        :param n_actions:
        :param lr:
        """
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope("Actor"):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name="l1"
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name="acts_prob"
            )

        with tf.variable_scope("exp_v"):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope("train"):
            # want exp_v as large as possible
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        """
        :param s: state
        :param a: action
        :param td: critic tells if the direction is right or wrong
        :return:
        """
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)

        return exp_v

    def choose_action(self, s):
        """
        Stochastically choose action according to state
        :param s:
        :return:
        """
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})

        return np.random.choice(np.arange(probs.shape[1]), p=probs.rave())


class ActorTarget(object):
    def __init__(self, sess, network, tau):
        self.sess = sess
        self.tau = tau

