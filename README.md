# UCD Reinforcement Learning
For this assignment, I used TensorFlow to train a convolutional neural network to learn to play Pong without human knowledge of the game. I used OpenAI Gym, which offered the game state as the game display (image), to run and interact with the Atari game. I also represented the game as a set of states, actions, and rewards to implement the Q-learning algorithm, and plotted loss and reward change during the training process.

`instructions.pdf` is the set of instructions with which I completed the assignment.  
`report3.pdf` is a written report of this assignment.

---

In previous assignments I manually created intelligent agents by codifying human intelligence regarding the environments my agent explored. However, modern approaches to artificial intelligence step away from methods requiring domain experts. This allows models to not only learn better policies, but also to learn policies for domains in which encoding human intelligence is near impossible.

One such domain is Pong, a tennis-themed arcade game, featuring simple two-dimensional graphics, in which each player operates a paddle and rallies the ball with the goal of making the opponent miss and earning a point. The game can be characterized as an environment for a reinforcement learning problem, with states, actions, and rewards: The state is the position of the paddles and the ball, the action is the direction and speed of paddle movement, and the reward is +1 or -1 depending on if the agent scores or is scored against, respectively (and 0 if neither occurs). Ultimately, through trial and error, an agent will learn a policy, or behavior function, that evaluates a given state and selects the action that maximizes its reward.

The model will use a deep Q-network, an algorithm enhancing classical Q-learning with deep learning and a technique called experience replay, to decide which action to take when playing Pong. A convolutional neural network will push each action made to a replay buffer; this replay buffer will be polled from to make updates to the deep Q-network using a loss function (squared error).

The deep Q-network is contained within `dqn.py`, and it is trained using `run_dqn_pong.py`. A single game can be played with the deep Q-network using `test_dqn_pong.py`. The environment will keep playing Pong until either player earns 21 points.

## Part 0: Environment and Setup
This may work locally with high-end GPU resources. Otherwise, you can use the Google Cloud platform and launch a Deep Learning VM:

> Google Cloud Platform, offered by Google, is a suite of cloud computing services that runs on the same infrastructure that Google uses internally for its end-user products, such as Google Search, Gmail, file storage, and YouTube: https://cloud.google.com/

If you are using Google Cloud Platform, here are some suggestions:
- When launching the Deep Learning VM, you can choose between TensorFlow and PyTorch frameworks. TensorFlow is preferred, because ApenAI Gym is installed by default.
- After launching the TensorFlow virtual machine, open its command line and do the following installation:  
```
sudo apt install cmake libz-dev
sudo pip install torch torchvision
sudo pip install gym[atari]
sudo apt-get install python-opengl
```
- Run your commands utilizing either `nohup` or `screen`. These will enable you to exit the terminal with your script still running.

Be aware that training this model will take several hours, even on high-end GPU servers! This is unavoidable.

## Part 1: Problem Representation
For the Q learner, we must represent the game as a set of states, actions, and rewards. OpenAI Gym offers two versions of game environments: one which offers the state as the game display (image) and one which offers the state as the hardware RAM (array). I explained why the former is easier for the agent to learn from.

I described the purpose of the neural network in Q learning. Since neural networks learn a complicated function mapping their inputs to their outputs, I explained what these variables (inputs and outputs) represent.

> The neural network in the provided code is `class QLearner(nn.Module)`

I described how an action is chosen:

```
if random.random() > epsilon:
    . . .
else:
    action = random.randrange(self.env.action_space.n)
```
<!-- 
`if random.random() > epsilon:`  
&nbsp;&nbsp;&nbsp;&nbsp;`. . .`  
`else:`  
&nbsp;&nbsp;&nbsp;&nbsp;`action = random.randrange(self.env.action_space.n)` -->

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
