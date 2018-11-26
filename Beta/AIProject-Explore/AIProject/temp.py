import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd,Adam
from itertools import product as possibleIterations
import matplotlib.pyplot as plt
import pickle


def savePickle(name, toSave):
    file = open(name, 'wb')
    pickle.dump(toSave, file)
    file.close()

def loadPickle(name):
    file = open(name, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def get_actions():
    possibleTorques = np.array([-1.0, 0.0, 1.0])
    possibleActions = np.array(
        list(possibleIterations(possibleTorques, possibleTorques, possibleTorques, possibleTorques)))
    return possibleActions


EPISODES = 1000
actions_space = get_actions()
model_file = 'pddqn'
def plot(data):
    x=[]
    y=[]
    for i,j in data:
        x.append(i)
        y.append(j)
    plt.plot(x,y)
    plt.savefig(model_file + '.png')

class State:
    def __init__(self, states, actions):
        self.states = states# 4 sattes
        self.actions = actions # 3 actions

    def get_input_layer(self):
        ret = []
        for i in range(3):
            ret = ret + list(self.states[i][0])
            # ret.append(actions_space.tolist().index(self.actions[i].tolist()))
            ret = ret + self.actions[i].tolist()
        ret = ret + list(self.states[3][0])
        # print("Before", ret)
        ret = np.array(ret)
        ret = np.reshape(ret, [1, 24*4 + 4*3])
        # print(ret.shape)
        # print(ret)
        return ret



class DeepQAgent:
    def __init__(self, state_size, action_space):
        self.state_size = state_size
        self.action_size = 81
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.95
        self.learning_rate = 1
        self.min_learning_rate = 0.1
        self.model = self._build_model()
        self.action_space = action_space
        self.priority_prob = deque(maxlen=20000)
        self.priority_bias = 0.001 # Tune this
        self.alpha = 0.6 # Importance of priority

    def get_action_from_prediction(self, predict):
        return self.action_space[np.argmax(predict[0])]

    def _build_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))  # changed layer count from 24
        model.add(Dense(80, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))  # changed this from linear
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, agent):
        action_index = self.action_space.tolist().index(action.tolist())
        self.memory.append((state, action_index, reward, next_state, done))
        self.update_probability(state, action, reward, next_state, done, agent)

    def update_probability(self, state, action, reward, next_state, done, agent):
        target = reward
        if not done:
            best_action = -1
            val = -100000000
            temp_val = self.model.predict(next_state.get_input_layer())[0]
            for a in range(len(self.action_space)):
                if temp_val[a] > val:
                    val = temp_val[a]
                    best_action = a
            target = (reward + self.gamma * agent.model.predict(next_state.get_input_layer())[0][best_action])
        td_error = target - self.model.predict(state.get_input_layer())[0][self.action_space.tolist().index(action.tolist())]
        priority = td_error + self.priority_bias
        priority = priority ** self.alpha
        self.priority_prob.append(priority)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space[np.random.choice([i for i in range(len(self.action_space))])]
        act_values = self.model.predict(state.get_input_layer())  # what does this return
        return self.get_action_from_prediction(act_values)

    def replay(self, batch_size, agent2):
        # minibatch = random.sample(self.memory, batch_size -1)

        minibatch = np.random.choice([i for i in range(len(self.memory))], size=batch_size-1, p = np.asarray(self.priority_prob)/sum(self.priority_prob))
        minibatch = [ self.memory[i] for i in minibatch ]
        minibatch += [self.memory[-1]]
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                best_action = -1
                val = -100000000
                temp_val = self.model.predict(next_state.get_input_layer())[0]
                for a in range(len(self.action_space)):
                    if temp_val[a] > val:
                        val = temp_val[a]
                        best_action = a
                '''check understading '''
                target = (reward + self.gamma * agent2.model.predict(next_state.get_input_layer())[0][best_action])  # Double Q learning
            target_f = self.model.predict(state.get_input_layer())
            target_f[0][action] = target
            '''check understading '''
            self.model.fit(state.get_input_layer(), target_f, epochs=1, verbose=0)

    def load(self, name):
        self.epsilon = self.epsilon_min
        self.learning_rate = self.min_learning_rate
        self.memory,self.priority_prob = loadPickle(name+"_memory")
        self.model.load_weights(name+".h5")

    def save(self, name):
        savePickle(name+"_memory",(self.memory,self.priority_prob))
        self.model.save_weights(name+".h5")


if __name__ == "__main__":
    eVSs = deque(maxlen=1000)
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape[0]
    action_size = len(actions_space)
    agent1 = DeepQAgent(state_size*4 + 4*3, actions_space)
    agent2 = DeepQAgent(state_size*4 + 4*3, actions_space)
    agent1.load(model_file + "agent1")
    agent2.load(model_file + "agent2")
    done = False
    batch_size = 32
    e = 0
    e,eVSs = loadPickle('iterNEW')
    recent_average = deque(maxlen=50)
    while True:
        e += 1
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
            # env.render()
            coin_toss = random.random() <= 0.5
            if coin_toss:
                action = agent1.act(curr_state)
            else:
                action = agent2.act(curr_state)

            my_actions.append(action)

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            my_state.append(state)

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}, alpha1: {}, alpha2: {}"
                      .format(e, EPISODES, total_reward, agent1.epsilon, agent2.learning_rate, agent1.learning_rate))
                recent_average.append(total_reward)
                av = sum(recent_average) / len(recent_average)
                print( " Recent Average = ", av)
                eVSs.append((e+1,av))
                break

            if len(my_state) == 4:
                curr_state = State(my_state, my_actions)
                if flag:
                    prev_state = curr_state
                    flag = False
                if coin_toss:
                    agent1.remember(prev_state, action, reward, curr_state, done, agent2)
                else:
                    agent2.remember(prev_state, action, reward, curr_state, done, agent1)

                prev_state = curr_state

            if coin_toss and len(agent1.memory) > batch_size:
                agent1.replay(batch_size, agent2)
            elif not coin_toss and len(agent2.memory) > batch_size:
                agent2.replay(batch_size, agent1)

        if agent1.epsilon > agent1.epsilon_min:
            agent1.epsilon *= agent1.epsilon_decay
        if agent1.learning_rate > agent1.min_learning_rate:
            agent1.learning_rate *= agent1.epsilon_decay

        if agent2.epsilon > agent2.epsilon_min:
            agent2.epsilon *= agent2.epsilon_decay
        if agent2.learning_rate > agent2.min_learning_rate:
            agent2.learning_rate *= agent2.epsilon_decay

        if e % 20 == 0:
            agent1.save(model_file + "agent1")
            agent2.save(model_file + "agent2")
            savePickle('iterNEW',(e,eVSs))
            plot(eVSs)