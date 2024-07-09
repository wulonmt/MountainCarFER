from gymnasium.envs.registration import register

register(
    id='MountainCarFixPos-v0',
    entry_point='Env.envs:MountainCarFixPos',
    max_episode_steps=200,
    reward_threshold=-110.0,
)
