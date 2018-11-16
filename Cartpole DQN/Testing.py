import gym
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[starty:starty+cropy, startx:startx+cropx, :]


def pre_process(obs):
    # img = obs[1:176:2, ::2]  # crop and downsize
    img = obs[::2, ::2]
    img = crop_center(img, 80, 80)
    img = img.sum(axis=2)  # to greyscale
    # img[img == 210 + 164 + 74] = 0  # Improve contrast
    # img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    img =  img.reshape(80, 80, 1)
    plt.imshow(img.reshape(80, 80), interpolation="nearest", cmap="gray")
    plt.show()


def create():
    env = gym.make("MsPacman-v0")
    obs = env.reset()
    print(obs.shape)
    print(env.action_space)
    # plt.imshow(obs)
    # plt.show()
    # img1 = Image.fromarray(obs, 'RGB')
    # img1.save('Init.png')
    pre_process(obs)


def main():
    env = gym.make('MsPacman-v0')
    score_db = []
    time_db = []
    for i_episode in range(500):
        observation = env.reset()
        total = 0
        for t in range(10000000):
            env.render()
            # print(observation)
            # time.sleep(0.1)
            print(env.action_space)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            total += reward
            if done:
                print("Episode ",i_episode," finished after {} timesteps".format(t + 1))
                print("Score = ", total)
                score_db.append(total)
                time_db.append(t+1)
                break
    env.env.close()
    print("Average score = ", sum(score_db)/500)
    print("Average survival time = ", sum(time_db)/500)


main()