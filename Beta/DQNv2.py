import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd
from itertools import product as possibleIterations

EPISODES = 1000

def get_actions():
    possibleTorques = np.array([-1.0, 0.0, 1.0])
    possibleActions = np.array(list(possibleIterations(possibleTorques, possibleTorques, possibleTorques, possibleTorques)))
    return possibleActions

class DeepQAgent:
    def __init__(self, state_size, action_space):
        self.state_size = state_size
        self.action_size = 81 # todo check, looks shady
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.action_space = actions_space

    def get_action_from_prediction(self, predict):
        return self.action_space[np.argmax(predict[0])]

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=sgd(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        action_index = self.action_space.tolist().index(action.tolist())
        self.memory.append((state, action_index, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon: # todo make this epsilon greedy
            return self.action_space[np.random.choice([i for i in range(len(self.action_space))])]
        act_values = self.model.predict(state) # what does this return
        return self.get_action_from_prediction(act_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0])) # Returns q-values
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape[0]
    actions_space = get_actions()
    action_size = len(actions_space)
    agent = DeepQAgent(state_size, actions_space)
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 50 == 0:
            agent.save("Bipedal-dqn-testing.h5")