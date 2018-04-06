import tensorflow as tf
from Actor import Actor, ActorTarget
from Critic import Critic, CriticTarget
from Neural_Network import NeuralNetworks
from ReplayBuffer import ReplayBuffer
import numpy as np
from MIMOEnv import MIMOEnv

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50
# Max episode length
MAX_EP_STEPS = 200
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.01
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001



# ===========================
#   Utility Parameters
# ===========================
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 128

# ===========================
#   Tensorflow Summary Ops
# ===========================


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================


def train(sess, env, network):
    arr_reward = np.zeros(MAX_EPISODES)
    arr_qmax = np.zeros(MAX_EPISODES)

    actor = Actor(sess, network, ACTOR_LEARNING_RATE)
    actor_target = ActorTarget(sess, network, TAU)
    critic = Critic(sess, network, CRITIC_LEARNING_RATE)
    critic_target = CriticTarget(sess, network, TAU)

    s_dim, a_dim, _ = network.get_const()

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    actor_target.train()
    critic_target.train()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    for i in range(MAX_EPISODES):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(MAX_EP_STEPS):

            # Added exploration noise
            a = actor.predict(np.reshape(s, (1, s_dim))) + (1. / (1. + i))

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (s_dim,)), np.reshape(a, (a_dim,)), r,
                              terminal, np.reshape(s2, (s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic_target.predict(s2_batch, actor_target.predict(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.resize(y_i, (MINIBATCH_SIZE, 1)))

                # ep_ave_max_q += np.amax(predicted_q_value)
                ep_ave_max_q += np.mean(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])


                # Update target networks
                actor_target.train()
                critic_target.train()

            s = s2
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                terminal = True

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                # writer.add_summary(summary_str, i)
                # writer.flush()

                print('Reward: ' + str(ep_reward) + ',   Episode: ' + str(i) + ',    Qmax: ' + str(
                    ep_ave_max_q / float(j)))
                arr_reward[i] = ep_reward
                arr_qmax[i] = ep_ave_max_q / float(j)

                # if i % 100 == 99:
                #     np.savez(RESULTS_FILE, arr_reward[0:i], arr_qmax[0:i])

                break


def main(_):
    with tf.Session() as sess:

        env = MIMOEnv()
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        # assert (env.action_space.high.all() == -env.action_space.low.all())

        network = NeuralNetworks(state_dim, action_dim, action_bound)

        train(sess, env, network)


if __name__ == '__main__':
    tf.app.run()
