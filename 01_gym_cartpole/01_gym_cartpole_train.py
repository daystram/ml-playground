import sys
sys.path.append('..')
import gym
from neuralnet import genetic_seq_neuralnet as nn

env = gym.make('CartPole-v1')
print(env.observation_space)
print(env.action_space)

env = nn.EnvWrapper(env, show=False)

shape = [(4,),          # input layer               
        (128, 'relu'),   # hidden layers
        (256, 'relu'), 
        (512, 'relu'), 
        (256, 'relu'), 
        (128, 'relu'), 
        (2, 'relu')]    # ouput layer

ga = nn.Generation(env, 20, 20, shape, step=10000, span=1, rate=0.3, verbose=True)
agent, _ = ga.run()

print("------ GA: Overall Best Reward {:.1f}".format(agent.reward))

nn.Agent.save(agent)
env.close()
