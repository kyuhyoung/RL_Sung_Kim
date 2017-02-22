import gym
import numpy as np
import tensorflow as tf
# Following is commented out due to the conflict between pyplot and cv2.imshow
'''
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
'''
from gym.envs.registration import register
import random as pr
import cv2
import sys
import math

dis = 0.99
learning_rate = 0.1
num_episode = 2000

def one_hot(x, dim):
    return np.identity(dim)[x:x + 1]

env = gym.make("FrozenLake-v0")

input_size = env.observation_space.n
output_size = env.action_space.n

X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))

Qpred = tf.matmul(X, W)

Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y - Qpred))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

rList = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episode):
        s = env.reset()
        e = 1. / ((i / 50) + 10)
        rAll = 0
        done = False
        local_loss = []
        while not done:
            Qs = sess.run(Qpred, feed_dict={X: one_hot(s, input_size)})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            s1, reward, done, _ = env.step(a)
            if done:
                Qs[0, a] = reward
            else:
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1, input_size)})
                Qs[0, a] = reward + dis * np.max(Qs1)
            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})
            rAll += reward
            s = s1
        rList.append(rAll)

