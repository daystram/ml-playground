import numpy as np
import pprint as p

# FF Neural Network Agent

def activation_unit(x):
    if x > 0:
        return 1
    else:
        return 0

unit = np.vectorize(activation_unit)

def activation_sign(x):
    if x > 0:
        return 1
    else:
        return -1

sign = np.vectorize(activation_sign)

def activation_relu(x):
    if x > 0:
        return x
    else:
        return 0

relu = np.vectorize(activation_relu)


class Agent:

    def __init__(self, shape, rate=0.5):
        self.reward = 0     # fitness

        self.shape = shape
        self.weights = []   # 2D
        self.bias = []      # 1D
        self.rate = rate

        for l, width in enumerate(self.shape[1:]):
            layer_weight = []
            layer_bias = []
            for node in range(width):
                layer_weight.append(np.random.uniform(-5, 5, self.shape[l]))
                layer_bias.append(np.random.uniform(-5, 5))
            self.weights.append(layer_weight)
            self.bias.append(layer_bias)
        # print('\n------ AGENT: weights, bias ------')
        # p.pprint(self.weights)
        # p.pprint(self.bias)
        
    def mutate(self, rate=None):
        if rate is None:
            rate = self.rate
        weights = []
        bias = []
        for l, width in enumerate(self.shape[1:]):
            layer_weight = []
            layer_bias = []
            for node in range(width):
                layer_weight.append(self.weights[l][node] + np.random.normal(0, rate, self.shape[l]))
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
        #TODO check observation width
        result = np.array(observation)

        for l, _ in enumerate(self.shape[1:]):
            result = relu(np.dot(self.weights[l], result) + self.bias[l])
        # print(unit(result))
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


class Generation:

    def __init__(self, wrapper, popSize, genCount, shape, rate=0.5):
        self.popSize = popSize
        self.genCount = genCount
        self.rate = rate
        self.population = [ Agent(shape, self.rate) for _ in range(self.popSize) ]
        self.env = wrapper
        self.best = None

    def run(self):
        print("------ GA: Starting")
        for gen in range(self.genCount):
            self.reset()
            self.simulate()
            self.select()
            self.debug(gen)
        return self.best
        
    def reset(self):
        for agent in self.population:
            agent.reset()
    
    def simulate(self, agent=None):
        if agent is None:
            for p, agent in enumerate(self.population):
                agent = self.env.execute(agent)
                if self.best is None or self.best.reward < agent.reward:
                    self.best = agent.copy()
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


class EnvWrapper:

    def __init__(self, env, render=False):
        self.env = env
        self.render = render

    def execute(self, agent, step=5000):
        observation = self.env.reset()
        s = 0
        while (s < step) or step == -1:
            if self.render: self.env.render()
            action = agent.act(observation)
            observation, reward, done, info = self.env.step(action)
            agent.award(reward)
            s += 1
            if done: break
        return agent

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()