import tensorflow as tf
from Actor import ActorNetwork
from Critic import CriticNetwork

from ReplayBuffer import ReplayBuffer
import numpy as np
from MIMOEnv import MIMOEnv
from DDPG import *



# Max training steps
MAX_EPISODES = 5000
# Max episode length
MAX_EP_STEPS = 200

# Soft target update param
TAU = 0.001

TEST = 10
ENV_NAME = "MIMOENV-v0"

def main():
    env = MIMOEnv()
    agent = DDPG(env)

    for episode in range(MAX_EPISODES):
        state = env.reset()
        # print("episode:", episode)
        total_reward = 0
        mean_q_value = 0
        # Train
        for step in range(MAX_EP_STEPS):
            action = agent.noise_action(state)
            next_state, reward, done, _ = env.step(action)
            q_value = agent.perceive(state, action, reward, next_state, done)
            mean_q_value += np.mean(q_value)

            state = next_state
            total_reward += reward
            if done:
                break
        print('episode: ', episode, 'Total Reward:', total_reward, 'Q value:', mean_q_value)

        # # Testing:
        # if episode % 10 == 0 and episode > 10:
        #     total_reward = 0
        #     for i in range(TEST):
        #         state = env.reset()
        #         for j in range(200):
        #             # env.render()
        #             action = agent.action(state)  # direct action for test
        #             state, reward, done, _ = env.step(action)
        #             total_reward += reward
        #             if done:
        #                 break
        #     ave_reward = total_reward / TEST
        #     print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    main()