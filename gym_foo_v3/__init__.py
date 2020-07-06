from gym.envs.registration import register

register(
    id='foo-v3',
    entry_point='gym_foo_v3.envs:FooEnv',
    max_episode_steps=3000,
    reward_threshold=2975.0,
)
