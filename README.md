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
Q-learning is the use of the Q-function of a policy to measure the expected sum of rewards, obtained from a given state, by taking an action and following the policy thereafter. For any finite Markov decision process, Q-learning finds an optimal policy by maximizing the expected value of the total reward (i.e. Q-value) over any and all successive states, starting from the current state. Since greater weight is given to more immediate rewards, future rewards are increasingly discounted.

To implement Q-learning, we must represent the game as a set of states, actions, and rewards. OpenAI Gym offers two versions of game environments: one which offers the state as the game display (image) and one which offers the state as the hardware RAM (array). I believe that the former would be easier for the agent to learn from, because the hardware RAM for an Atari 2600 was only 128 bytes, and it is questionable whether or not this is enough information to learn from.

Neural networks learn a complex function mapping inputs to outputs. Thus, the purpose of the convolutional neural network in this deep Q-network is to incorporate function approximation algorithms (i.e. backpropagation); a neural network substitutes for the Q-table, or table of Q-values for each action in each state, and uses each Q-function update as a training example. The convolutional neural network is trained with the game display, or the state, as input and six Q-functions, or the possible actions, as output.  
The deep Q-network in the provided code is defined by `class QLearner(nn.Module)`.

The following code in `dqn.py` affects how an action is chosen:
```
if random.random() > epsilon:
    . . .
else:
    action = random.randrange(self.env.action_space.n)
```
There are two strategies for selecting an action to execute in the current state: exploration and exploitation. The agent either explores new paths to rewards by randomly selecting an action (with probability `epsilon`), or exploits what it has learned so far by selecting the action corresponding to the largest Q-value (with probability 1 - `epsilon`). Therefore, how frequently either strategy is used is determined by the value of `epsilon`. Ideally, the value of `epsilon` decreases during training, because exploration is preferred at the beginning and exploitation is preferred at the end. The Q-learner only decides what action to perform when using the exploitation strategy.

Given a state, I wrote code in `dqn.py` to compute the Q-value and choose an action to perform.

## Part 2: Making Q-Learner Learn
The objective function of the deep Q-network for one-state lookahead is to iteratively reduce the discrepancy between Q-value estimates for adjacent states. In doing so, it uses the squared error loss function:

<img src="https://render.githubusercontent.com/render/math?math=Loss_i(\Theta_i)=(y_i-Q(s,a,\Theta_i))^2">

- <img src="https://render.githubusercontent.com/render/math?math=\Theta_i"> - neural network weights for iteration *i*
- <img src="https://render.githubusercontent.com/render/math?math=y_i"> - target Q-value for iteration *i*
- <img src="https://render.githubusercontent.com/render/math?math=Q(s,a,\Theta_i))"> - predicted Q-value for iteration *i*
    - The Q-value is the reward received immediately upon applying action *a* to state *s*, plus the value (discounted by <img src="https://render.githubusercontent.com/render/math?math=\gamma">) of following the optimal policy thereafter.

This loss function helps our model learn because, under certain assumptions, it converges to the optimal Q-function (and, unlike supervised learning, the targets are not provided and fixed beforehand).

## Part 3: Extend Deep Q-Learner
The replay memory/buffer in deep Q-networks is used to store many (state, action, reward, next state) entries. In typical Q-learning, these entries are used sequentially to update the Q-table. With experience replay, we instead store these entries and later use them to update our policy by randomly sampling from the buffer to get an entry and using that entry to update our policy. This is necessary as optimization is assuming independent and identically-distributed samples. If we do not use the experience replay, then the agent will see many similar samples, as seqential samples in the game are very similar. This will encourage the model to converge to a local minimum.

I implemented the "random sampling" function of the replay memory/buffer in `dqn.py`:
```
def sample(self, batch_size):
    iterables = random.sample(self.buffer, batch_size)
    state, action, reward, next_state, done = zip(*iterables)
    
    return state, action, reward, next_state, done
```

## Part 4: Learning to Play Pong
I began with `model_pretrained.pth`, a partially-trained model that was to be loaded for further training. Since it is good convention when training neural networks to save your model occasionally (in case code crashes, servers are disrupted, and so on), I adjusted `run_dqn_pong.py` to be able to load in a model and occasionally save a model to disk.
> There are built-in functions to do so in PyTorch, such as `torch.save(model.state_dict(), filename)`. Do not manually extract or load in the model weights. Use `.pth` files.

I trained the model by running `run_dqn_pong.py`. To achieve good performance, I needed to train the model for approximately 500,000 more frames (which takes 3 to 6 hours on Google servers). While optimizing different values for hyperparameters such as `epsilon` and the size of the replay buffer were an option, I decided not to do so, as it was unnecessary. My final model is saved in `model.pth`

`run_dqn_pong.py` recorded the loss and rewards in `losses` and `all_rewards`, respectively. I modified `run_dqn_pong.py` to save these to memory (in `losses.txt` and `all_rewards.txt`). Then, I used a Python script, `plot_graphs.py`, to use the data in `losses.txt` and `all_rewards.txt` to plot how the loss and reward changed during the training process:

<img src="https://i.postimg.cc/zDcMRqc5/loss.png" width="768" height="384">

<img src="https://i.postimg.cc/nc9W4Nym/reward.png" width="768" height="384">

These figures were also included in my report.
