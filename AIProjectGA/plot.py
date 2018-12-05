import numpy as np
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


# In[27]:

def plot(data,name):
    x=[]
    y=[]
    for i in data:
        x.append(i[0])
        y.append(i[1:])
    y = np.array(y)
    for i in range(len(y[0])):
        plt.plot(x,y[:,i:i+1])
    plt.plot(x,y)
    plt.savefig(name + '.png')

plot(loadPickle('plotData'),'plot.png')