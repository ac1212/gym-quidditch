from gym.envs.registration import register

register(
    id='quidditch-snitch-v0',
    entry_point='gym_quidditch.envs:QuidditchSnitchEnv',
)