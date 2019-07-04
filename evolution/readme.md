# CartPole

## Model

A simple neural network was used to solve this problem (Figure 1). It is fully connected (with bias) 
and uses the ReLU activation function throughout, except for the output layer which uses sigmoid. 
The input size of 4 corresponds to the 4 observations CartPole produces at each step. 
CartPole has two possible actions, so network output less than 0.5 was interpreted as action 0 (move left) 
and output greater than or eqal to 0.5 was interpreted as action 1 (move right).

<p align="center">
  <img src="https://raw.githubusercontent.com/ConorLazarou/ai-gym/master/evolution/cart_pole/examples/network.png" width="200">
</p>
<p align="center">
  Figure 1: Network Architecture
</p>

## Training
100 random neural networks with the above architecture were generated. 
Each network was applied to CartPole 10 times, until failure or 500 steps had passed.
Each network was scored based on the number of steps that passed before the game ended, with a higher score being better.
The 90 worst networks were deleted.
The 10 best networks were each duplicated 9 times, with small gaussian noise added to each weight of each duplicate.
The new networks were again applied to CartPole and the entire process was repeated a total of 20 times.
The best network after 20 generations was kept, and is illustrated in figure 2.
An small penalty was added for the cart deviating from the centre of the track and for the pole deviating from vertical;
Training was repeated with this penalty, and the best network after 20 generations is illustrated in figure 3.

## Examples
<p align="center">
  <img src="https://raw.githubusercontent.com/ConorLazarou/ai-gym/master/evolution/cart_pole/examples/default.gif" width="400">
</p>
<p align="center">
  Figure 2: The best-performing network playing CartPole
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ConorLazarou/ai-gym/master/evolution/cart_pole/examples/stability.gif" width="400">
</p>
<p align="center">
  Figure 3: The best-performing network trained with the deviance penalty playing CartPole
</p>
