import sys
sys.path.append('..')
import gym
from neuralnet import genetic_seq_neuralnet as nn

env = gym.make('MountainCar-v0')
print(env.observation_space)
print(env.action_space)

env = nn.EnvWrapper(env, show=False)

shape = [(2,),          # input layer               
        (32, 'tanh'),   # hidden layers
        (64, 'tanh'), 
        (32, 'tanh'), 
        (3, 'relu')]    # ouput layer

ga = nn.Generation(env, 10, 10, shape)
agent = ga.run()

print("------ GA: Overall Best Reward {:.1f}".format(agent.reward))

nn.Agent.save(agent)
env.close()
