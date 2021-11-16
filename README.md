# UCD Reinforcement Learning
In previous assignments I manually created intelligent agents by codifying human intelligence regarding the environments my agent explored. However, modern approaches to artificial intelligence step away from methods requiring domain experts. This allows models to not only learn better policies, but also to learn policies for domains in which encoding human intelligence is near impossible.

For this assignment, I used TensorFlow to train a convolutional neural network to learn to play Pong without human knowledge of the game. I used OpenAI Gym, which offered the game state as the game display (image), to run and interact with the Atari game. I also represented the game as a set of states, actions, and rewards to implement the Q-learning algorithm, and plotted loss and reward change during the training process.

This may work locally with high-end GPU resources. Otherwise, you can use the Google Cloud platform and launch a Deep Learning VM:

> Google Cloud Platform, offered by Google, is a suite of cloud computing services that runs on the same infrastructure that Google uses internally for its end-user products, such as Google Search, Gmail, file storage, and YouTube: https://cloud.google.com/

## Part 1: Problem Representation
For the Q learner, we must represent the game as a set of states, actions, and rewards. OpenAI Gym offers two versions of game environments: one which offers the state as the game display (image) and one which offers the state as the hardware RAM (array). I explained why the former is easier for the agent to learn from.

I described the purpose of the neural network in Q learning. Since neural networks learn a complicated function mapping their inputs to their outputs, I explained what these variables (inputs and outputs) represent.

> The neural network in the provided code is `class QLearner(nn.Module)`

I described how an action is chosen:

`if random.random() > epsilon:`  
&nbsp;&nbsp;&nbsp;&nbsp;`. . .`  
`else:`  
&nbsp;&nbsp;&nbsp;&nbsp;`action = random.randrange(self.env.action_space.n`

Given a state, I wrote code to compute the Q value and choose an action to perform (see lines 50-55 in function `act` of **dqn.py**).

## Part 2: Making Q-Learner Learn
I explained the objective function of deep Q-learning neural networks for one-state lookahead.

Loss function (mean squared error):  
<img src="https://render.githubusercontent.com/render/math?math=Loss_i(\Theta_i)=(y_i-Q(s,a,\Theta_i))^2">

## Part 3: Extend Deep Q-Learner
The replay memory/buffer in deep Q networks is used to store many (state, action, reward, next state) entries. In typical Q learning, these entries are used sequentially to update the Q table. With experience replay, we instead store these entries and later use them to update our policy by randomly sampling from the buffer to get an entry and using that entry to update our policy. This is necessary as optimization is assuming independent and identically distributed samples. If we do not use the experience replay then the agent will see many similar samples as seqential samples in the game are very similar. This will encourage the model to converge to a local minimum.

I implemened the "random sampling" function of replay memory/buffer. This samples a batch from the replay buffer (see line 90, function `sample` of **dqn.py**).

## Part 4: Learning to Play Pong
I began with a partially trained network that was to be loaded for further training. Since it is good convention when training neural networks to save your model occasionally, I adjusted **run_dqn_pong.py** to be able to load in a model and occasionally save a model to disk.

**run_dqn_pong.py** recorded the loss and rewards in `losses` and `all_rewards` respectively. I modified **run_dqn_pong.py** to save these to memory.

I trained the model by running **run_dqn_pong.py**. To achieve good performance, I needed to train the model for approximately 500,000 more frames (which takes 3 to 6 hours on Google servers). Hyperparameters such as epsilon and the size of the replay buffer can be optimized.

I plotted how the loss and reward changed during the training process, and I included these figures in my report.

### Submission
- **report3.pdf**
- **model.pth**: My best-performing model saved to a `.pth` file; **test_dqn_pong.py** can load my saved model.
- **run_dqn_pong.py**
- **dqn.py**
