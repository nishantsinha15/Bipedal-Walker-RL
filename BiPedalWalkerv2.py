#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import numpy as np
import tensorflow as tf
from itertools import product as possibleIterations


# In[ ]:


class BipedalWalkerModel:
    def __init__(self):
        self.env = gym.make("BipedalWalker-v2")
        self.obs = self.env.reset()
        possibleTorques = np.array([-1.0, 0.0, 1.0])
        self.possibleActions = np.array(list(possibleIterations(possibleTorques, possibleTorques, possibleTorques, possibleTorques)))
        print(self.possibleActions.shape)
        self.initNetworkGraph()
        
    def initNetworkGraph(self, learningRate = 0.01):
        self.nInputLayer = self.env.observation_space.shape[0]  #24
        nHiddenLayer1 = 20
        nHiddenLayer2 = 40
        nOutputLayer = len(self.possibleActions) #81
        initializer = tf.variance_scaling_initializer()
        
        self.X = tf.placeholder(tf.float32, shape=[None, self.nInputLayer])
        hidden1 = tf.layers.dense(self.X, nHiddenLayer1, activation=tf.nn.selu, kernel_initializer=initializer)
        hidden2 = tf.layers.dense(hidden1, nHiddenLayer2, activation=tf.nn.selu, kernel_initializer=initializer)
        logits = tf.layers.dense(hidden2, nOutputLayer, kernel_initializer=initializer)
        outputs = tf.nn.softmax(logits)
        
        self.logitIndex = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=-1)
        y = tf.one_hot(self.logitIndex, depth=len(self.possibleActions))
        crossEntropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        optimizer = tf.train.AdamOptimizer(learningRate)
        
        gradientsAndVariables = optimizer.compute_gradients(crossEntropy)
        self.gradients = [g for g,v in gradientsAndVariables]        
        self.gradientPlaceholders = []
        gradientsandVariableFeedDict = []
        for grad, variable in gradientsAndVariables:
            gradientPlaceholder = tf.placeholder(tf.float32, shape=grad.get_shape())
            self.gradientPlaceholders.append(gradientPlaceholder)
            gradientsandVariableFeedDict.append((gradientPlaceholder, variable))
        self.train = optimizer.apply_gradients(gradientsandVariableFeedDict)
        self.saver = tf.train.Saver()
        
    def trainNetwork(self, Iterations = 1000, killAfterSteps = 1000, batchSize = 10, renderEnv = False):
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            for iteration in range(Iterations):
                print("\rIteration: {}/{}".format(iteration + 1, Iterations), end="")
                allRewards = []
                allGradients = []
                for game in range(batchSize):
                    currentRewards = []
                    currentGradients = []
                    obs = self.env.reset()
                    for step in range(killAfterSteps):
                        if renderEnv:
                            self.env.render()
                        actionIndex, gradientsValue = session.run([self.logitIndex, self.gradients], feed_dict={self.X: obs.reshape(1, self.nInputLayer)})
                        action = self.possibleActions[actionIndex]
                        obs, reward, done, info = self.env.step(action[0])
                        currentRewards.append(reward)
                        currentGradients.append(gradientsValue)
                        if done:
                            break
                    allRewards.append(currentRewards)
                    allGradients.append(currentGradients)
                
                allRewards = self.processRewards(allRewards, rate=0.95)
                feed_dict = {}
                for i, gradientPlaceholder in enumerate(self.gradientPlaceholders):
                    newGradients = [reward * allGradients[gameIndex][step][i]
                                      for gameIndex, rewards in enumerate(allRewards)
                                          for step, reward in enumerate(rewards)]
                    meanGradients = np.mean(newGradients, axis=0)
                    feed_dict[gradientPlaceholder] = meanGradients
                session.run(self.train, feed_dict=feed_dict)
                if iteration % 10 == 0:
                    self.saver.save(session, "./model.ckpt")
        
        
        
    def propagateFinalRewardBackward(self, allRewards, rate ):
        finalRewards = np.zeros(len(allRewards))
        cumulativeRewards = 0
        for step in reversed(range(len(allRewards))):
            cumulativeRewards = allRewards[step] + cumulativeRewards * rate
            finalRewards[step] = cumulativeRewards
        return finalRewards
    
    def normalizeRewards(self, allRewards):
        flattenedRewards = np.concatenate(allRewards)
        rewardMean = flattenedRewards.mean()
        rewardStd = flattenedRewards.std()
        normalizedRewards =  [(reward - rewardMean)/rewardStd for reward in allRewards]
        return normalizedRewards
    
    def processRewards(self, allRewards, rate = 0.8):
        propagatedRewards = [self.propagateFinalRewardBackward(rewards, rate) for rewards in allRewards]
        normalizedRewards = self.normalizeRewards(propagatedRewards)
        return normalizedRewards
    
    def run(self, model_path = "./model.ckpt", maxSteps = 1000 ):
        env = gym.make("BipedalWalker-v2")
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            obs = self.env.reset()
            for step in range(maxSteps):
                self.env.render(mode="rgb_array")
                action_index_val = self.logitIndex.eval(feed_dict={self.X: obs.reshape(1, self.nInputLayer)})
                action = self.possibleActions[action_index_val]
                obs, reward, done, info = self.env.step(action[0])
                if done:
                    break
        self.env.close()
        
        


# In[ ]:


myModel = BipedalWalkerModel()


# In[ ]:


myModel.trainNetwork(renderEnv = False)


# In[ ]:


myModel.run()


# In[ ]:




