import gym

env = gym.make('gym_quidditch:quidditch-snitch-v0')
env.reset()

for i_episode in range(5):
    observation = env.reset()
    for t in range(1000):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()