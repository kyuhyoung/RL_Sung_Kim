# -*- coding: utf-8 -*-
import gym
import readchar


LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT }

env = gym.make("FrozenLake-v0")
#env._episode_started_at = 0
env.reset()
env.render()


while True:
    #key = inkey()
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print('Game aborted !!')
        break
    action = arrow_keys[key]
    #print('action key input : ' + str(action))
    #print('env._episode_started_at : ' + str(env._episode_started_at))
    state, reward, done, info = env.step(action)
    env.render()
    print('State : ', state, "Action : ", action, "Reward : ", reward, "Info : ", info)
    if done:
        print("Finished with reward", reward)
        break