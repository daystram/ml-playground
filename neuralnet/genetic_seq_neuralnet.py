import numpy as np
import pprint as p
import pickle
import time

# Activation Methods
def activation_unit(x):
    if x > 0:
        return 1
    else:
        return 0

def activation_sign(x):
    if x > 0:
        return 1
    else:
        return -1

def activation_relu(x):
    if x > 0:
        return x
    else:
        return 0

def activation_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activation_tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

activation = {
    'unit': np.vectorize(activation_unit),
    'sign': np.vectorize(activation_sign),
    'relu': np.vectorize(activation_relu),
    'sigmoid': np.vectorize(activation_sigmoid),
    'tanh': np.vectorize(activation_tanh)
}


# FF Neural Network Agent
class Agent:

    def __init__(self, shape, rate=0.5, span=1):
        self.reward = 0     # fitness

        self.shape = shape
        self.weights = []   # 2D
        self.bias = []      # 1D
        self.rate = rate

        for l, layer in enumerate(self.shape[1:]):
            layer_weight = []
            layer_bias = []
            for node in range(layer[0]):
                layer_weight.append(np.random.uniform(-span, span, self.shape[l][0]))
                layer_bias.append(np.random.uniform(-span, span))
            self.weights.append(layer_weight)
            self.bias.append(layer_bias)
        
    def mutate(self, rate=None):
        if rate is None:
            rate = self.rate
        weights = []
        bias = []
        for l, layer in enumerate(self.shape[1:]):
            layer_weight = []
            layer_bias = []
            for node in range(layer[0]):
                layer_weight.append(self.weights[l][node] + np.random.normal(0, rate, self.shape[l][0]))
                layer_bias.append(self.bias[l][node] + np.random.normal(0, rate))
            weights.append(layer_weight)
            bias.append(layer_bias)
        return weights[:], bias[:]

    def child(self, rate=None):
        if rate is None:
            rate = self.rate

        child = Agent(self.shape[:], rate)
        child.weights, child.bias = self.mutate()

        return child

    def act(self, observation):
        result = np.array(observation)
        for l, layer in enumerate(self.shape[1:]):
            result = activation[layer[1]](np.dot(self.weights[l], result) + self.bias[l])
        return np.argmax(result)

    def award(self, reward):
        self.reward += reward
    
    def reset(self):
        self.reward = 0

    def copy(self):
        copy = Agent(self.shape[:], self.rate)
        copy.reward = self.reward
        copy.weights = self.weights
        copy.bias = self.bias
        return copy
    
    @staticmethod
    def save(agent, filename=None):
        if filename is None:
            filename = 'agent-{}.model'.format(time.strftime("%Y%m%d_%H%M%S"))
        file = open(filename, 'wb')
        pickle.dump(agent, file=file)
        file.close()

    @staticmethod
    def load(filename):
        file = open(filename, 'rb')
        agent = pickle.load(file)
        file.close()
        return agent


# Genetic Algorithm Runner
class Generation:

    def __init__(self, wrapper, popSize, genCount, shape, rate=0.5, span=1, verbose=False, step=1000):
        self.popSize = popSize
        self.genCount = genCount
        self.rate = rate
        self.span = span
        self.verbose = verbose
        self.population = [ Agent(shape, self.rate, self.span) for _ in range(self.popSize) ]
        self.env = wrapper
        self.best = None
        self.step = step

    def run(self):
        print("------ GA: Starting")
        for gen in range(self.genCount):
            self.reset()
            self.simulate()
            self.select()
            if self.verbose: self.debug(gen)
        return self.best
        
    def reset(self):
        for agent in self.population:
            agent.reset()
    
    def simulate(self, agent=None):
        if agent is None:
            for p, agent in enumerate(self.population):
                agent = self.env.execute(agent, self.step)
                if self.best is None or self.best.reward < agent.reward:
                    self.best = agent.copy()
                if self.verbose: print("--- Agent #{:<2d}: Reward {:3.1f}".format(p, agent.reward))
        else:
            return self.env.execute(agent)

    def select(self, ratio=0.25, rate=None):
        subdivision = [int(ratio * self.popSize), self.popSize - int(ratio * self.popSize)]

        prob = np.array([ agent.reward for agent in self.population ])
        prob /= prob.sum()

        selected = np.random.choice(self.population, size=subdivision[0], replace=False, p=prob)

        prob = np.array([ agent.reward for agent in selected ])
        prob /= prob.sum()

        if rate is None:
            rate = self.rate

        self.population = list(selected) + [ agent.child() for agent in np.random.choice(selected, size=subdivision[1], p=prob) ]

        best = self.population[0]
        for agent in self.population:
            if agent.reward > best.reward:
                best = agent
        if self.best is None or best.reward > self.best.reward:
            self.best = best

    def debug(self, gen):
        best = self.population[0]
        for agent in self.population:
            if best.reward < agent.reward:
                best = agent
        print("Generation {:2d}: {:.1f}".format(gen, best.reward))


# Environment Wrapper
class EnvWrapper:

    def __init__(self, env, show=False):
        self.env = env
        self.show = show

    def execute(self, agent, step=1000):
        agent.reset()
        observation = self.env.reset()
        s = 0
        while (s < step) or step == -1:
            action = agent.act(observation)
            if self.show: 
                self.env.render()
            observation, reward, done, _ = self.env.step(action)
            agent.award(reward)
            s += 1
            if done: break
        return agent

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()
