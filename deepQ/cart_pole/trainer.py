import gym
import agent.py

episodes = 1000
max_time = 500

if __name__ == "__main__":

	#initialize
	env = gym.make("CartPole-v0")
	agent = DQNAgent(env)


	# iterate the game and reset it
	for e in range(episodes):

		#reset the state each time
		state = env.reset()
		state = np.reshape(state, [1, 4])

		#run game for a fixed amount of time
		for t in range(max_time)

			# take an action for each step
			action = agent.act(state)

			# get the reward for the step
			next_state, reward, done, _ = env.step(action)
			next_state = np.reshape(next_state, [1, 4])

			agent.remember(state, action, reward, next_state, done)

			state = next_state


			# if the simulation finished this round, print score and exit
			if done:
				print("Episode: {}/{}, Score: {}".format(e, episodes, t))

				break




		# replay the agent to train it
		agent.replay(32)