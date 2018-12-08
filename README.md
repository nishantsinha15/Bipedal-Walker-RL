# Artificial-Intelligence-Course-Project
Used Deep Q Learning, Double DQN, Prioritized DDQN, Deep Recurrent Q-Network and Genetic Algorithm to solve the problem. 
The best score is achieved using Genetic Algorithm.

# About the Environment 

https://gym.openai.com/envs/BipedalWalker-v2/

Making a bipedal walk
Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.
