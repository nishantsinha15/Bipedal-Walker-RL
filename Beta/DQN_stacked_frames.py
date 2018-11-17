import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd
from itertools import product as possibleIterations
import matplotlib.pyplot as plt


def get_actions():
    possibleTorques = np.array([-1.0, 0.0, 1.0])
    possibleActions = np.array(
        list(possibleIterations(possibleTorques, possibleTorques, possibleTorques, possibleTorques)))
    return possibleActions


EPISODES = 1000
actions_space = get_actions()

def plot(data):
    x=[]
    y=[]
    for i,j in data:
        x.append(i)
        y.append(j)
    plt.plot(x,y)
    plt.savefig('temp.png')

class State:
    def __init__(self, states, actions):
        self.states = states# 4 sattes
        self.actions = actions # 3 actions

    def get_input_layer(self):
        ret = []
        for i in range(3):
            ret = ret + list(self.states[i][0])
            ret.append(actions_space.tolist().index(self.actions[i].tolist()))
        ret = ret + list(self.states[3][0])
        ret = np.array(ret)
        ret = np.reshape(ret, [1, 24*4 + 3])

        return ret



class DeepQAgent:
    def __init__(self, state_size, action_space):
        self.state_size = state_size
        self.action_size = 81
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.action_space = action_space

    def get_action_from_prediction(self, predict):
        return self.action_space[np.argmax(predict[0])]

    def _build_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))  # changed layer count from 24
        model.add(Dense(80, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # changed this from linear
        model.compile(loss='mse', optimizer=sgd(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        action_index = self.action_space.tolist().index(action.tolist())
        self.memory.append((state, action_index, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space[np.random.choice([i for i in range(len(self.action_space))])]
        act_values = self.model.predict(state.get_input_layer())  # what does this return
        return self.get_action_from_prediction(act_values)

    def replay(self, batch_size, agent2):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(agent2.model.predict(next_state.get_input_layer())[0]))  # Returns q-values
            target_f = agent2.model.predict(state.get_input_layer())
            target_f[0][action] = target
            self.model.fit(state.get_input_layer(), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    eVSs = deque(maxlen=1000)
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape[0]
    action_size = len(actions_space)
    agent = DeepQAgent(state_size*4 + 3, actions_space)
    agent2 = DeepQAgent(state_size*4 + 3, actions_space)
    done = False
    batch_size = 32
    c = 0
    recent_average = deque(maxlen=100)
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        prev_state = State([state for i in range(4)], [np.array([-1.0, 0.0, 1.0, 1.0]) for i in range(3)])
        curr_state = State([state for i in range(4)], [np.array([-1.0, 0.0, 1.0, 1.0]) for i in range(3)])
        my_state = deque(maxlen=4)
        my_actions = deque(maxlen=3)
        my_state.append(state)
        flag = True
        for time in range(500):
            c += 1
            # env.render()
            action = agent.act(curr_state)
            my_actions.append(action)

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            my_state.append(state)

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, total_reward, agent.epsilon))
                recent_average.append(total_reward)
                av = sum(recent_average) / len(recent_average)
                print("c = ", c, " Recent Average = ", av)
                eVSs.append((e+1,av))
                break

            if len(my_state) == 4:
                curr_state = State(my_state, my_actions)
                if flag:
                    prev_state = curr_state
                    flag = False
                agent.remember(prev_state, action, reward, curr_state, done)
                prev_state = curr_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size, agent2)
            if c >= 1000:
                print('updating model')
                c = 0
                agent2.model.set_weights(agent.model.get_weights())
        if e % 1 == 0:
            agent.save("Bipedal-dqn-nishant-v2.h5")
            plot(eVSs)

