import sys
sys.path.append('..')
import gym
from neuralnet import genetic_seq_neuralnet as nn

env = nn.EnvWrapper(gym.make('CartPole-v1'), show=True)

agent = nn.Agent.load(sys.argv[1])
print("------ Agent: Reward {:.1f}".format(agent.reward))
for _ in range(10):
    env.execute(agent, -1)

env.close()
