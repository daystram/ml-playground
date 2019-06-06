import gym
import genetic_ff_neuralnet as nn

env = nn.EnvWrapper(gym.make('CartPole-v1'), render=False)

ga = nn.Generation(env, 20, 20, [4, 128, 256, 512, 256, 128, 2])
agent = ga.run()

print("------ GA: Overall Best Reward {:.1f}".format(agent.reward))
env.render = True
env.execute(agent, -1)

env.close()