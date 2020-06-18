import random

from vwgym import VacuumWorldPO


env = VacuumWorldPO(10)
state = env.reset()
print(env.action_space)


print(state.center)


for i in range(10):
    state, reward, done, *_ = env.step(random.randint(0,env.action_space.n-1))
    print(state.center)