
# coding: utf-8

# In[1]:


# import os

# def installWrapper():
#     # Install a Drive FUSE wrapper.
#     # https://github.com/astrada/google-drive-ocamlfuse
#     !apt-get update -qq 2>&1 > /dev/null
#     !apt-get install -y -qq software-properties-common python-software-properties module-init-tools
#     !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
#     !apt-get update -qq 2>&1 > /dev/null
#     !apt-get -y install -qq google-drive-ocamlfuse fuse
    
# def auth():
#     from google.colab import auth
#     auth.authenticate_user()
    
# def authorizeWrapper():
#     # Generate creds for the Drive FUSE library.
#     from google.colab import output
#     from oauth2client.client import GoogleCredentials
#     import time
#     creds = GoogleCredentials.get_application_default()
#     import getpass
#     # Determine if Drive Fuse credential setup is already complete.
#     fuse_credentials_configured = False
#     with output.temporary():
#       !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1
#       # _exit_code is set to the result of the last "!" command.
#       fuse_credentials_configured = _exit_code == 0
#     # Sleep for a short period to ensure that the previous output has been cleared.
#     time.sleep(1)
#     if fuse_credentials_configured:
#       print('Drive FUSE credentials already configured!')
#     else:
#       # Work around misordering of STREAM and STDIN in Jupyter.
#       # https://github.com/jupyter/notebook/issues/3159
#       prompt = !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
#       vcode = getpass.getpass(prompt[0] + '\n\nEnter verification code: ')
#       !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
    
# def createDriveDir():
#     !mkdir -p drive
#     !google-drive-ocamlfuse drive
    
# installWrapper()
# auth()
# authorizeWrapper()
# createDriveDir()


# In[ ]:


import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd,Adam
from itertools import product as possibleIterations
import copy


# In[ ]:


class DeepQAgent:
    def __init__(self, state_size, action_space):
        self.state_size = state_size
        self.action_size = 81 # todo check, looks shady
        self.memory = deque(maxlen=200000) # chaged this from 2000
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05 # changed this from 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.action_space = actions_space

    def get_action_from_prediction(self, predict):
        return self.action_space[np.argmax(predict[0])]

    def _build_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu')) # changed layer count from 24
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.action_size, activation='relu')) # changed this from linear
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        action_index = self.action_space.tolist().index(action.tolist())
        self.memory.append((state, action_index, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
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


# In[ ]:


EPISODES = 1000

def get_actions():
    possibleTorques = np.array([-1.0, 0.0, 1.0])
    possibleActions = np.array(list(possibleIterations(possibleTorques, possibleTorques, possibleTorques, possibleTorques)))
    return possibleActions

if __name__ == "__main__":
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape[0]
    actions_space = get_actions()
    action_size = len(actions_space)
    agent = DeepQAgent(state_size, actions_space)
    # agent2 = DeepQAgent(state_size, actions_space)
    done = False
    batch_size = 32
    c = 0
    recent_average = deque(maxlen = 100)
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(500):
            c += 1
            #env.render()
            action = agent.act(state)
            print(action)
            next_state, reward, done, _ = env.step(action)
            print(next_state, reward)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, total_reward, agent.epsilon))
                recent_average.append(total_reward)
                print("c = ", c, " Recent Average = ", sum(recent_average)/len(recent_average))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            # if c >= 1000:
            #     print('updating model')
            #     c = 0
            #     # agent2 = copy.deepcopy(agent)
            #     agent2.model.set_weights(agent.model.get_weights())
        if e % 50 == 0:
            agent.save("Bipedal-dqn-testing.h5")

