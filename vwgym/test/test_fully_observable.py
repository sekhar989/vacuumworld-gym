import random

from vwgym import VacuumWorld, Vectorise


env = VacuumWorld(3)
env = Vectorise(env)
state = env.reset()
print(env.action_space)


for i in range(10):
    state, reward, done, *_ = env.step(random.randint(0,env.action_space.n-1))
    print(state)