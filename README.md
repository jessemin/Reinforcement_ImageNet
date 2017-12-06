# Reinforcement ImageNet
## Double DQN to Bounding Box Problem

In this work, we suggest a double deep Q-network agent that detects human from images.
There already have been several RL-based approach toward image classification and localization problem.
(Juan C Caicedo et al., Miriam Bellver et al.)
However, in this work, we try to improve the performance by adapting double DQN with pre-trained VGG-16.
We first define object detection task as a Markov decision process with state, action, and reward.
The agent then learns how to transform a bounding box using simple transformation actions to reach the reasonable bounding box that captures human from given images.
The agent is trained and evaluated on the ImageNet bounding box annotation dataset and compared with simpler agents based on deep Q-network model with and without experience replay.
The results show that double deep Q-network model with experience replay outperforms models based on vanilla deep Q-network.
We also show that we can infer what the double DQN agent learned by tracking sequences of actions the agent take.
This is a final Project for Stanford CS238: Decision under uncertainty.
