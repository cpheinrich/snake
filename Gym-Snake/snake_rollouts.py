import gym
import gym_snake

env = gym.make('snake-plural-v0')
env.n_snakes = 24
env.grid_size = [80,80]
env.unit_size = 5
env.unit_gap = 1
env.n_foods = 200

for i in range(100):
    env.reset()
    for t in range(1000):
        env.render()
        actions = []
        for i in range(env.n_snakes):
            actions.append(env.action_space.sample())
        observation, reward, done, info = env.step(actions)
        if done:
            print('episode {} finished after {} timesteps'.format(i, t))
            break
