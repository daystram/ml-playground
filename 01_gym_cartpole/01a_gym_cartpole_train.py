import sys
sys.path.append('..')
import gym
from neuralnet import genetic_seq_neuralnet as nn

env = gym.make('CartPole-v1')
print(env.observation_space)
print(env.action_space)

env = nn.EnvWrapper(env, show=False)

shape = [(4,),          # input layer               
        (8, 'relu'),   # hidden layers
        (16, 'relu'), 
        (8, 'relu'), 
        (2, 'relu')]    # ouput layer

ga = nn.Generation(env, 20, 20, shape, step=10000, span=1, rate=0.3, verbose=False)
agent, _ = ga.run()

print("------ GA: Overall Best Reward {:.1f}".format(agent.reward))

nn.Agent.save(agent)
env.close()
