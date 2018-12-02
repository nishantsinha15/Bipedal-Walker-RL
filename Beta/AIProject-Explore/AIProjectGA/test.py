import random
import gym
import numpy as np
import heapq

env = gym.make('BipedalWalker-v2')

class Chromosome:
    def __init__(self, evaluate = True):
        self.deadPoint = None
        if evaluate:
            self.data = self.getRandomGene()
            self.score = self.evaluate()
        else:
            self.data = None
            self.score = None


    def getRandomGene(self):
        return np.random.rand(1600,4)

    def evaluate(self):
        score = 0
        #print(len(self.data))
        env.reset()
        #print(state)
        for i in range(len(self.data)):
            _, reward, done, _ = env.step(self.data[i])
            score += reward
            if done:
                self.deadPoint = i
                break
        if self.deadPoint == None:
            self.deadPoint = 1600
        return score

    def twoPointCrossOver(self, other):
        if self.deadPoint == 1600:
            x = random.randint(0,len(self.data)-2)
        elif self.deadPoint > 15:
            x = self.deadPoint - 15
        y = random.randint(x, len(self.data)-1)

        child1 = Chromosome()
        child1.data = self.data
        child1.data[x:y] = other.data[x:y]
        child1.score = child1.evaluate()

        return [child1]
            

def GA():
    chromosomes = [Chromosome() for i in range(100)]
    e = 0
    while True:
        chromosomes.sort(key=lambda x: x.score)
        print("iteration: ", e, " | best Score: ", chromosomes[-1].score, " | worst Score: ", chromosomes[0].score)
        chromosomes = chromosomes[-10:]
        newChromosomes = []
        for i in range(len(chromosomes)):
            for j in range(len(chromosomes)):
                newChromosomes += chromosomes[i].twoPointCrossOver(chromosomes[j])
        chromosomes += newChromosomes
        for i in chromosomes:
            print(i.score, i.deadPoint)
        e+=1




GA()


