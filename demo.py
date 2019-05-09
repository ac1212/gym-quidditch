import gym

GENERATE_FRAMES = False
if GENERATE_FRAMES:
    import scipy.misc
render_mode = 'rgb_array' if GENERATE_FRAMES else 'human'

env = gym.make('gym_quidditch:quidditch-snitch-v0')
env.reset()

for i_episode in range(5):
    observation = env.reset()
    for t in range(1000):
        rgb = env.render(mode=render_mode)
        if GENERATE_FRAMES:
            scipy.misc.imsave('frames/frames_e{:04d}-t{:04d}.png'.format(i_episode,t),rgb)
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()