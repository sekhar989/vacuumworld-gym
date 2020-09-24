## Required Installations
___

This project was built using conda package manager and environment management system. Both `conda` and `pip` installation instructions are mentioned in the following sections.
Before proceeding please make sure to install the headless vacuumworld-gym according to the instructions mentioned [here](https://github.com/sekhar989/vacuumworld-gym#vacuumworld-gym). Post installation please update the `__init__.py` using this [script](https://github.com/sekhar989/vacuumworld-gym/blob/master/vwgym/__init__.py). Also, make sure to have the [`init_utils.py`](https://github.com/sekhar989/vacuumworld-gym/blob/master/vwgym/init_utils.py) in the same folder.

1. Install PyTorch  

Please refer to the [PyTorch](https://pytorch.org/) documentation for installation instructions according to your system configuration.  

Recommended installation for GPU systems:

- `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`
  
Recommended installation for CPU systems:  
    
- `conda install pytorch torchvision cpuonly -c pytorch`

2. Other Required Packages  

Other necessary packages can be installed in one of the following ways depending on the type of environment management system used:
- Conda
    - `while read requirement; do conda install --yes $requirement; done < requirements.txt`
- Python pip
    - `pip3 install -r requirements.txt`

## Folder Structure
---

Agents trained in a `3 x 3` grid with `6-dirt locations` arranged using the random seed `518123` are saved inside the `saved_model` folder present under each main folder. The folder structure is shown below.

- [trained_agents](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents)
    - [A2C](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents/A2C)  
        - [a2c_train_logs](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents/A2C/a2c_train_logs)  
        - [saved_model](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents/A2C/saved_model)  
            - `actor.pt`  
            - `critic.pt`  
        - `actor_critic_a2c.py`  
        - `a2c_train.py`
        - `actor-critic-test.ipynb`
    - [DQN](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents/DQN)
        - [dqn_train_logs](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents/DQN/dqn_train_logs)
        - [saved_model](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents/DQN/saved_model)
            - `dqn_ckpt.pt`
        - `dqn.py`
        - `dqn_train.py`
        - `dqn-test.ipynb`
    - [FuN-Lite](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents/FuN-Lite)
        - [hrl_train_logs](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents/FuN-Lite/hrl_train_logs)
        - .[saved_model](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents/FuN-Lite/saved_model)
            - `fnet_ckpt.pt`
        - `fun_lite.py`
        - `hrl_train.py`
        - [goal_analysis](https://github.com/sekhar989/vacuumworld-gym/tree/master/trained_agents/FuN-Lite/goal_analysis)
        - `f_net_test.ipynb`
        
To run the pre-trained models please run the corresponding `<algorithm>_test.ipynb` notebooks.

## Training
---

To train any agent from scratch, please navigate to the corresponding agent type directory and run the `<algorithm>_train.py` script. This repository has implementations of traditional RL algorithms including Q-Learning, advanced (Deep) RL algorithms including [DQN](https://arxiv.org/abs/1312.5602) and [Actor-Critic](https://arxiv.org/abs/1602.01783), and an implementation of a deep hierarchical RL algorithm called [Feudal Networks (FuN)](https://arxiv.org/abs/1703.01161).

By default it will run on a `3 x 3` grid size with `6-dirt locations` arranged with random seed `518123`. To train your own configuration, please run the corresponding agent type (algorithm) script in the with corresponding arguments as shown below:

`python hrl_train.py --grid_size=5 --sparse=1 --seed=1`  
`python a2c_train.py --grid_size=5 --sparse=1 --seed=1`  
`python dqn_train.py --grid_size=5 --sparse=1 --seed=1`

`sparse = 1` will put twice the number of dirt locations in the environment whereas `sparse = 0` will put only three dirt locations irrespective of the grid_size. The corresponding training process can be observed on tensorboard by using the below command:  
`tensorboard --logdir hrl_train_logs/`  
`tensorboard --logdir a2c_train_logs/`  
`tensorboard --logdir dqn_train_logs/`  

This helps us to monitor the training process in real-time.

## Test
---

To test the trained agent the parameters which were used to train should be used in the corresponding test notebook. In `cell-2` of every test notebook, the environment is declared. If there’s change in the parameter during training other than the default parameter, it has to also reflect here.

`env, input_shape = make_env(grid_size=3, num_env=1, vectorize=True)`   
`env = env[0]`  
`env.state()`  
