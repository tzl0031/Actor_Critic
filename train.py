from Actor import Actor
from Critic import Critic
from ReplayBuffer import ReplayBuffer
import tensorflow as tf

MAX_EPISODE = 3000
MAX_TIME = 200

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# initialize Actor, Critic ,Buffer
actor = Actor(sess, n_features=2, n_actions=3, lr=0.001)
critic = Critic(sess, n_features=2, lr=0.01)
replay_buffer = ReplayBuffer(10000)

if __name__ == '__main__':
    for i in range(MAX_EPISODE):
        done = False
        state
        track_r = []
        t = 0
        for j in range(MAX_TIME):
            # current state
            s =
            # choose an action randomly
            a = actor.choose_action(s)
            # Actor returns an action ut
            # execute ut
            # s_ new state
            s_ =
            # reward
            r =
            # store the tuple in RM
            # sample a mini batch of n from RM
            # compute



            td_error = critic.learn(s, r, s_)
            actor.learn(s, a, td_error)

            s = s_
            t += 1
