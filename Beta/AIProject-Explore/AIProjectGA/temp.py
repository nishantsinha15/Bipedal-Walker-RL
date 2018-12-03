import time, math, random, bisect, copy
import gym
import numpy as np
import pickle


GAME = 'BipedalWalker-v2'
MAX_STEPS = 1600
MAX_GENERATIONS = 1000
POPULATION_COUNT = 100
MUTATION_RATE = 0.01

def savePickle(name, toSave):
    file = open(name, 'wb')
    pickle.dump(toSave, file)
    file.close()

def loadPickle(name):
    file = open(name, 'rb')
    data = pickle.load(file)
    file.close()
    return data



class Network: 
    def __init__(self, layers):     
        self.fitnessScore = 0.0
        self.layers = layers
        self.nodeWeights = []
        self.nodeBiases = []
        for i in range(len(layers) - 1):
            self.nodeWeights.append( np.random.uniform(low=-1, high=1, size=(layers[i], layers[i+1])).tolist() )
            self.nodeBiases.append( np.random.uniform(low=-1, high=1, size=(layers[i+1])).tolist())
  
    def getAction(self, input):
        action = input
        for i in range(len(self.layers)-1):
            action = np.reshape( np.matmul(action, self.nodeWeights[i]) + self.nodeBiases[i], (self.layers[i+1]))
        return action

    def evaluate(self):
        observation = env.reset()
        totalReward = 0
        for _ in range(MAX_STEPS):
            action = self.getAction(observation)
            observation, reward, done, _ = env.step(action)
            totalReward += reward
            if done:
                break
        self.fitnessScore = totalReward


class Population :
    def __init__(self, populationCount, mutationRate, layers):
        self.layers = layers
        self.populationCount = populationCount
        self.mutationRate = mutationRate
        self.population = [ Network(layers) for i in range(self.populationCount)]



    def createChild(self, network1, network2):
        child = Network(self.layers)
        for i in range(len(child.nodeWeights)):
            for j in range(len(child.nodeWeights[i])):
                for k in range(len(child.nodeWeights[i][j])):
                    if random.random() > self.mutationRate:
                        if random.random() < network1.fitnessScore / (network1.fitnessScore+network2.fitnessScore):
                            child.nodeWeights[i][j][k] = network1.nodeWeights[i][j][k]
                        else :
                            child.nodeWeights[i][j][k] = network2.nodeWeights[i][j][k]
        for i in range(len(child.nodeBiases)):
            for j in range(len(child.nodeBiases[i])):
                if random.random() > self.mutationRate:
                    if random.random() < network1.fitnessScore / (network1.fitnessScore+network2.fitnessScore):
                        child.nodeBiases[i][j] = network1.nodeBiases[i][j]
                    else:
                        child.nodeBiases[i][j] = network2.nodeBiases[i][j]

        return child



env = gym.make(GAME)
layers =  [env.observation_space.shape[0], 100, 80, 25, env.action_space.shape[0]]
pool = Population(POPULATION_COUNT, MUTATION_RATE,layers)

bestNetworks = []
plotData =[]
for gen in range(MAX_GENERATIONS):
    averageFitness = 0.0
    minFitness =  10000000
    maxFitness = -10000000
    maxNetwork = None

    # evaluation
    for nn in pool.population:
        observation = env.reset()
        totalReward = 0
        for step in range(MAX_STEPS):
            #env.render()
            action = nn.getAction(observation)
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                break

        nn.fitnessScore = totalReward
        minFitness = min(minFitness, nn.fitnessScore)
        averageFitness += nn.fitnessScore
        if nn.fitnessScore > maxFitness :
            maxFitness = nn.fitnessScore
            maxNetwork = copy.deepcopy(nn)
            savePickle('bestWeight',maxNetwork)

    bestNetworks.append(maxNetwork)
    averageFitness/=pool.populationCount
    print("Generation : ",gen+1," | Av : ", averageFitness, " | Max: ", maxFitness)

    # new generation creation
    nextGeneration = []
    randomIntroductions = [ Network(pool.layers) for i in range(10)]
    for t in randomIntroductions:
        t.evaluate()
    pool.population += randomIntroductions
    pool.population.sort(key=lambda x: x.fitnessScore, reverse=True)
    for i in range(pool.populationCount):
        if random.random() < float(pool.populationCount-i)/pool.populationCount:
            nextGeneration.append(copy.deepcopy(pool.population[i]))

    fitnessSum = [0]
    minFitness = min([i.fitnessScore for i in nextGeneration])
    for i in range(len(nextGeneration)):
        fitnessSum.append(fitnessSum[i]+(nextGeneration[i].fitnessScore-minFitness)**4)
    
    while(len(nextGeneration) < pool.populationCount):
        random1 = random.uniform(0, fitnessSum[-1] )
        random2 = random.uniform(0, fitnessSum[-1] )
        index1 = bisect.bisect_left(fitnessSum, random1)
        index2 = bisect.bisect_left(fitnessSum, random2)
        if 0 <= index1 < len(nextGeneration) and 0 <= index2 < len(nextGeneration) :
            nextGeneration.append( pool.createChild(nextGeneration[index1], nextGeneration[index2]) )
    pool.population = nextGeneration


    plotData.append([gen+1, averageFitness, maxFitness])
    savePickle('plotData',plotData)


# nn = loadPickle('bestWeightBackup')

# for _ in range(10):
#     observation = env.reset()
#     totalReward = 0
#     for step in range(MAX_STEPS):
#         action = nn.getAction(observation)
#         observation, reward, done, info = env.step(action)
#         totalReward += reward
#         if done:
#             break

#     print(totalReward)





