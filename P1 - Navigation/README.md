[//]: # (Image References)


[video_random]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Random Agent"


[video_trained]: https://github.com/gianscarpe/udacity_deep_reinforcement_learning/blob/master/P1%20-%20Navigation/contents/agent.gif "Trained Agent"

# Project 1: Navigation

| Random agent             |  Trained agent |
:-------------------------:|:-------------------------:
![Random Agent][video_random]  |  ![Trained Agent][video_trained]


### Introduction
This is my solution of the project "Navigation" from Udactiy Nanodegree on Deep
Reinforcement Learning. I suggest to read the documentation (and take the
course) before checking my solution.

.### Environment

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/`
   folder, and unzip (or decompress) the file. 
   
3. Install conda environment with `conda env create -f environment.yml`

### Description
- `Navigation.ipynb`: I present my own implementation of Deep Q-learning
  algorithm (DQN). I encourage to check out the original DQN paper for
  reference. [[Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)]
  
### Results
Plot showing the score per episode over all the episodes. The environment was
solved in **270** episodes. 

[dqn-scores](https://raw.githubusercontent.com/dalmia/udacity-deep-reinforcement-learning/master/2%20-%20Value-based%20methods/Project-Navigation/results/dddqn_new_scores.png)


