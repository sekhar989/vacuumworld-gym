# vacuumworld-gym
Vacuum World as a Gym environment

### local install:

```
git clone git@github.com:BenedictWilkinsAI/vacuumworld-gym.git
cd vacuumworld-gym
pip install -e .
```
### use

```
import random
from vwgym import VacuumWorld, Vectorise


env = VacuumWorld(3)
env = Vectorise(env)
state = env.reset()

for i in range(10):
    action = random.randint(0,env.action_space.n-1)
    state, reward, done, *_ = env.step(action)
```
For running instructions please refer [here](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents).
