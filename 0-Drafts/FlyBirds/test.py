import gym
from gym import envs

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
print(sorted(env_ids))

env = gym.make("Pong-v0")

observation = env.reset()

print(observation)


render = True

if render:
    env.render()
# from ale_py import ALEInterface
# from ale_py.roms import SpaceInvaders

# ale = ALEInterface()
# ale.loadROM(SpaceInvaders)