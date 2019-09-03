import gym
import numpy as np
from agent import DQNAgent
import os, time

episodes = 5000
max_time = 500
gamma = 0.9
batch_size = 8


def test_model(filename):
	env = gym.make("CartPole-v1")
	agent = DQNAgent(4, 2)
	agent.load_model(filename)

	state = env.reset()

	for _ in range(1000):
		env.render()
		state, _, _, _ = env.step(agent.act(state, explore=False))

	env.close()


if __name__ == "__main__":

	base_time = int(time.time())

	if not os.path.exists(f"{os.getcwd()}/models"):
		os.mkdir(f"{os.getcwd()}/models")

	#initialize
	env = gym.make("CartPole-v1")
	agent = DQNAgent(4, 2)


	# iterate the game and reset it
	running_t = 0
	for e in range(1, episodes + 1):

		#reset the state each time
		state = env.reset()

		#run game for a fixed amount of time
		for t in range(1, max_time + 1):

			# take an action for each step
			action = agent.act(state)

			# get the reward for the step
			next_state, reward, done, _ = env.step(action)

			agent.remember(state, action, reward, next_state, done)

			state = next_state


			# if the simulation finished this round, print score and exit
			if done:
				running_t = running_t * gamma + t * (1-gamma)
				print(f"Episode: {e}/{episodes}, Score: {t}, smoothed: {int(np.round(running_t))}")

				break


		# replay the agent to train it
		agent.replay(batch_size)


		if e % 1000 == 0:
			agent.save_model(f"{os.getcwd()}/models/cartpole_{base_time}_{e}.h5")



	agent.save_model(f"{os.getcwd()}/models/cartpole_{base_time}_final.h5")