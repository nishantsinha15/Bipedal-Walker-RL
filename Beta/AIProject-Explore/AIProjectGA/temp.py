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



class NeuralNet: 
    def __init__(self, layers):     
        self.fitness = 0.0
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            self.weights.append( np.random.uniform(low=-1, high=1, size=(layers[i], layers[i+1])).tolist() )
            self.biases.append( np.random.uniform(low=-1, high=1, size=(layers[i+1])).tolist())
  
    def getOutput(self, input):
        output = input
        for i in range(len(self.layers)-1):
            output = np.reshape( np.matmul(output, self.weights[i]) + self.biases[i], (self.layers[i+1]))
        return output

    def evaluate(self):
        observation = env.reset()
        totalReward = 0
        for _ in range(MAX_STEPS):
            action = self.getOutput(observation)
            observation, reward, done, _ = env.step(action)
            totalReward += reward
            if done:
                break
        self.fitness = totalReward


class Population :
    def __init__(self, populationCount, mutationRate, layers):
        self.layers = layers
        self.popCount = populationCount
        self.m_rate = mutationRate
        self.population = [ NeuralNet(layers) for i in range(populationCount)]



    def createChild(self, nn1, nn2):
        child = NeuralNet(self.layers)
        for i in range(len(child.weights)):
            for j in range(len(child.weights[i])):
                for k in range(len(child.weights[i][j])):
                    if random.random() > self.m_rate:
                        if random.random() < nn1.fitness / (nn1.fitness+nn2.fitness):
                            child.weights[i][j][k] = nn1.weights[i][j][k]
                        else :
                            child.weights[i][j][k] = nn2.weights[i][j][k]
                        

        for i in range(len(child.biases)):
            for j in range(len(child.biases[i])):
                if random.random() > self.m_rate:
                    if random.random() < nn1.fitness / (nn1.fitness+nn2.fitness):
                        child.biases[i][j] = nn1.biases[i][j]
                    else:
                        child.biases[i][j] = nn2.biases[i][j]

        return child


    def createNewGeneration(self):    
        nextGen = []
        randomIntroductions = [ NeuralNet(self.layers) for i in range(10)]
        for t in randomIntroductions:
            t.evaluate()
        self.population += randomIntroductions
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(self.popCount):
            if random.random() < float(self.popCount-i)/self.popCount:
                nextGen.append(copy.deepcopy(self.population[i]))

        fitnessSum = [0]
        minFit = min([i.fitness for i in nextGen])
        for i in range(len(nextGen)):
            fitnessSum.append(fitnessSum[i]+(nextGen[i].fitness-minFit)**4)
        
        while(len(nextGen) < self.popCount):
            r1 = random.uniform(0, fitnessSum[-1] )
            r2 = random.uniform(0, fitnessSum[-1] )
            i1 = bisect.bisect_left(fitnessSum, r1)
            i2 = bisect.bisect_left(fitnessSum, r2)
            if 0 <= i1 < len(nextGen) and 0 <= i2 < len(nextGen) :
                nextGen.append( self.createChild(nextGen[i1], nextGen[i2]) )
        self.population = nextGen




def replayBestBots(bestNeuralNets, steps, sleep):  
    choice = input("Do you want to watch the replay ?[Y/N] : ")
    if choice=='Y' or choice=='y':
        for i in range(len(bestNeuralNets)):
            if (i+1)%steps == 0 :
                observation = env.reset()
                totalReward = 0
                for step in range(MAX_STEPS):
                    env.render()
                    time.sleep(sleep)
                    action = bestNeuralNets[i].getOutput(observation)
                    observation, reward, done, info = env.step(action)
                    totalReward += reward
                    if done:
                        observation = env.reset()
                        break
                print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))




env = gym.make(GAME)
observation = env.reset()
in_dimen = env.observation_space.shape[0]
out_dimen = env.action_space.shape[0]
layers =  [in_dimen, 13, 8, 13, out_dimen]
pop = Population(POPULATION_COUNT, MUTATION_RATE,layers)
bestNeuralNets = []
plotData =[]


# for gen in range(MAX_GENERATIONS):
#     genAvgFit = 0.0
#     minFit =  1000000
#     maxFit = -1000000
#     maxNeuralNet = None
#     for nn in pop.population:
#         observation = env.reset()
#         totalReward = 0
#         for step in range(MAX_STEPS):
#             #env.render()
#             action = nn.getOutput(observation)
#             observation, reward, done, info = env.step(action)
#             totalReward += reward
#             if done:
#                 break

#         nn.fitness = totalReward
#         minFit = min(minFit, nn.fitness)
#         genAvgFit += nn.fitness
#         if nn.fitness > maxFit :
#             maxFit = nn.fitness
#             maxNeuralNet = copy.deepcopy(nn)
#             savePickle('bestWeight',maxNeuralNet)

#     bestNeuralNets.append(maxNeuralNet)
#     genAvgFit/=pop.popCount
#     print("Generation : %3d  |  Min : %5.0f  |  Avg : %5.0f  |  Max : %5.0f  " % (gen+1, minFit, genAvgFit, maxFit) )
#     pop.createNewGeneration()
#     plotData.append([gen+1, genAvgFit, maxFit])
#     savePickle('plotData',plotData)


nn = loadPickle('bestWeight')

for _ in range(10):
    observation = env.reset()
    totalReward = 0
    for step in range(MAX_STEPS):
        action = nn.getOutput(observation)
        observation, reward, done, info = env.step(action)
        totalReward += reward
        if done:
            break

    print(totalReward)





