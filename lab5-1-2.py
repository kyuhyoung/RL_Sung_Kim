import gym
import numpy as np
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

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
side_row, side_col = 100, 100
fontFace = cv2.FONT_HERSHEY_COMPLEX
fontScaleQ = 0.005 * min(side_row, side_col)
fontScaleEpisode = 0.006 * min(side_row, side_col)
thicTextQ = int(fontScaleQ)
tipSize = 0.15 * min(side_row, side_col)
row_text = int(side_row * 0.8)
x_shift, y_shift = side_col * 0.04, side_row * 0.04
#showTime = 50
showTime = 10
num_episodes = 2000
dis = .99
n_slip = 0
n_move = 0


def draw_q(im, x, y, i, q):

    if UP == i or LEFT == i:
        a = 0
    y_offset, x_offset = side_row * 0.05, -side_col * 0.17
    if i == LEFT:
        x_offset = -side_col * 0.45
    elif i == RIGHT:
        x_offset = side_col * 0.11
    elif i == UP:
        y_offset = -side_row * 0.28
    else:
        y_offset = side_row * 0.4
    x_txt, y_txt = int(x + x_offset), int(y + y_offset)
    text = '%1.2f' % (q)
    cv2.putText(im, text, (x_txt, y_txt), fontFace, fontScaleQ, 0, thicTextQ)
    return im

def draw_reward(im, i_epi, rew, rList):
    rate_success = str(sum(rList) / num_episodes)
    h, w, ch = im.shape
    text = 'epi.:' + str(i_epi) + '/' + str(num_episodes) + '  rew.:' + str(rew) + '  s.r.:' + str(rate_success)
    x_txt = int(w * 0.01)
    y_txt = int(h - row_text * 0.6)
    cv2.putText(im, text, (x_txt, y_txt), fontFace, fontScaleEpisode, 0)
    if n_move:
        rate_slip = float(n_slip) / float(n_move)
    else:
        rate_slip = 0
    text = 'slip rate : ' +  '%1.2f' % (rate_slip)
    y_txt = int(h - row_text * 0.25)
    cv2.putText(im, text, (x_txt, y_txt), fontFace, fontScaleEpisode, 0)
    #cv2.imshow('temp', im)
    #cv2.waitKey()
    return im


def checkQ(i_state):
    if 14 == i_state:
        a = 0


def draw_initial(n_row, n_col, Q, i_epi, rew, rList, li_hole):
    n_state, n_action = Q.shape
    #nn = Q.shape
    if n_row * n_col != n_state:
        print('check # row and # col')
        sys.exit()
    h, w = n_row * side_row + row_text, n_col * side_col
    im_lake = 255 * np.ones((h, w, 3), np.uint8)
    for r in range(1, n_row + 1):
        y = r * side_row
        cv2.line(im_lake, (0, y), (w, y), 0)
    for c in range(1, n_col):
        x = c * side_col
        cv2.line(im_lake, (x, 0), (x, h - row_text), 0)
    for r in range(n_row):
        y_up = r * side_row
        y_down = y_up + side_row
        for c in range(n_col):
            x_left = c * side_col
            x_right = x_left + side_col
            cv2.line(im_lake, (x_left, y_up), (x_right, y_down), 0)
            cv2.line(im_lake, (x_right, y_up), (x_left, y_down), 0)
            i_state = r * n_col + c
            x_center, y_center = 0.5 * (x_left + x_right), 0.5 * (y_up + y_down)
            for i_action in range(n_action):
                q = Q[i_state, i_action]
                if q:
                    checkQ(i_state)
                    im_lake = draw_q(im_lake, x_center, y_center, i_action, q)
            if i_state in li_hole:
                im_lake = draw_hole(im_lake, x_left, x_right, y_up, y_down)
    im_lake = draw_reward(im_lake, i_epi, rew, rList)

    #cv2.imshow('lake', im_lake)
    #cv2.waitKey(showTime)
    return im_lake

def draw_arrow(im, x_from, y_from, x_to, y_to, kolor, thick_line):
    if x_from == x_to and y_from == y_to:
        return im
    if x_from < x_to:
        y_from, y_to = y_from - y_shift, y_to - y_shift
    elif x_from > x_to:
        y_from, y_to = y_from + y_shift, y_to + y_shift

    if y_from < y_to:
        x_from, x_to = x_from - x_shift, x_to - x_shift
    elif y_from > y_to:
        x_from, x_to = x_from + x_shift, x_to + x_shift

    cv2.line(im, (int(x_from), int(y_from)), (int(x_to), int(y_to)), kolor, thick_line);
    angle_deg = cv2.fastAtan2(y_to - y_from, x_to - x_from)
    angle_rad = angle_deg * math.pi / 180.0
    for sain in range(-1, 2, 2):
        x_tip = round(x_to - tipSize * math.cos(angle_rad + sain * math.pi / 8))
        y_tip = round(y_to - tipSize * math.sin(angle_rad + sain * math.pi / 8))
        cv2.line(im, (int(x_tip), int(y_tip)), (int(x_to), int(y_to)), kolor, thick_line)
    return im

def draw_hole(im, x_l, x_r, y_u, y_d):
    cv2.line(im, (x_l, y_u), (x_r, y_u), 0, 2)
    cv2.line(im, (x_l, y_d), (x_r, y_d), 0, 2)
    cv2.line(im, (x_l, y_u), (x_l, y_d), 0, 2)
    cv2.line(im, (x_r, y_u), (x_r, y_d), 0, 2)
    cv2.line(im, (x_l, y_u), (x_r, y_d), 0, 2)
    cv2.line(im, (x_l, y_d), (x_r, y_u), 0, 2)
    return im

def is_slipped(state, action, new_state):
    is_slip = False
    if UP == action:
        if 0 <= state and state <= 3:
            is_slip = state != new_state
        else:
            is_slip = state - 4 != new_state
    elif DOWN == action:
        if 12 <= state and state <= 15:
            is_slip = state != new_state
        else:
            is_slip = state + 4 != new_state
    elif LEFT == action:
        if 0 == state % 4:
            is_slip = state != new_state
        else:
            is_slip = state - 1 != new_state
    else:
        if 3 == state % 4:
            is_slip = state != new_state
        else:
            is_slip = state + 1 != new_state
    return is_slip




def draw_path(im, state, action, new_state, n_col, reward, done):
    c_pre = state % n_col
    r_pre = (state - c_pre) / n_col
    x_l_pre = c_pre * side_col
    x_r_pre = x_l_pre + side_col
    y_u_pre = r_pre * side_row
    y_d_pre = y_u_pre + side_row
    x_c_pre, y_c_pre = 0.5 * (x_l_pre + x_r_pre), 0.5 * (y_u_pre + y_d_pre)
    c_nex = new_state % n_col
    r_nex = (new_state - c_nex) / n_col
    x_l_nex = c_nex * side_col
    x_r_nex = x_l_nex + side_col
    y_u_nex = r_nex * side_row
    y_d_nex = y_u_nex + side_row
    x_c_nex, y_c_nex = 0.5 * (x_l_nex + x_r_nex), 0.5 * (y_u_nex + y_d_nex)
    if (not reward) and done:
        im = draw_hole(im, x_l_nex, x_r_nex, y_u_nex, y_d_nex)
    global n_move
    n_move += 1
    if is_slipped(state, action, new_state):
        im = draw_arrow(im, x_c_pre, y_c_pre, x_c_nex, y_c_nex, (0, 0, 255), 2)
        global n_slip
        n_slip += 1
    else:
        im = draw_arrow(im, x_c_pre, y_c_pre, x_c_nex, y_c_nex, (0, 0, 0), 2)
    return im



def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

env = gym.make("FrozenLake-v0")

Q = np.zeros([env.observation_space.n, env.action_space.n])

li_hole = []
rList = []
r_all_pre = 0
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    im_lake = draw_initial(4, 4, Q, i - 1, r_all_pre, rList, li_hole)
    while not done:
        action = np.argmax(Q[state, :] + np.random.rand(1, env.action_space.n) / (i + 1))
        #action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
        new_state, reward, done, _ = env.step(action)
        is_slip = is_slipped(state, action, new_state)
        if reward:
            checkQ(state)
        q_pre = Q[state, action]
        Q[state, action] = reward + dis * np.max(Q[new_state, :])
        q_nex = Q[state, action]
        is_Q_changed = q_pre != q_nex
        rAll += reward
        im_lake = draw_path(im_lake, state, action, new_state, 4, reward, done)
        state = new_state
        if done and 0 == reward:
            if new_state not in li_hole:
                li_hole.append(new_state)
        cv2.imshow('lake', im_lake)
        cv2.waitKey(showTime)
    rList.append(rAll)
    r_all_pre = rAll

print('Success rate : ', str(sum(rList)/num_episodes))
print('Final Q-Table values')
print('Left Down Right Up')
print(Q)
# Following is commented out due to the conflict between pyplot and cv2.imshow
#plt.bar(range(len(rList)), rList, color='blue')
#plt.show()
