import gym
import genetic_ff_neuralnet as nn

env = nn.EnvWrapper(gym.make('CartPole-v1'), render=False)

shape = [(4,),          # input layer               
        (128, 'relu'),  # hidden layers
        (256, 'relu'), 
        (512, 'relu'), 
        (256, 'relu'), 
        (128, 'relu'), 
        (2, 'unit')]    # ouput layer

ga = nn.Generation(env, 20, 20, shape)
agent = ga.run()

print("------ GA: Overall Best Reward {:.1f}".format(agent.reward))
env.render = True
env.execute(agent, -1)

env.close()