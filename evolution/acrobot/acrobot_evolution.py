import numpy as np
import gym
import copy
import sys

ENV_NAME = 'Acrobot-v1'
LIFETIME = 120

class NN():
	def __init__(self):
		self.FC1 = np.random.normal(0,np.sqrt(2/(6+32)),(6,32))
		self.bias1 = np.random.normal(0,np.sqrt(2/(6+32)),(1,32))
		self.FC2 = np.random.normal(0,np.sqrt(2/(32+16)),(32,16))
		self.bias2 = np.random.normal(0,np.sqrt(2/(32+16)),(1,16))
		self.FC3 = np.random.normal(0,np.sqrt(2/(16+3)),(16,3))
		self.bias3 = np.random.normal(0,np.sqrt(2/(16+3)),(1,3))


	def relu(self, X):
		return X * (X>=0)

	def sigmoid(self, X):
		return 1 / (1 + np.exp(-X)) 

	def softmax(self, X):
		return np.exp(X) / np.sum(np.exp(X))

	def predict_proba(self, X):
		X = np.array(X).reshape((-1,6))
		X = X @ self.FC1 + self.bias1
		X = self.relu(X)
		X = X @ self.FC2 + self.bias2
		X = self.relu(X)
		X = X @ self.FC3 + self.bias3
		X = self.softmax(X)
		return X

	def predict(self, X):
		X = self.predict_proba(X)
		return np.argmax(X, axis=1).reshape((-1, 1))


	def mutate(self, stdev=0.03):
		self.FC1 += np.random.normal(0, stdev, self.FC1.shape)
		self.FC2 += np.random.normal(0, stdev, self.FC2.shape)
		self.FC3 += np.random.normal(0, stdev, self.FC3.shape)
		self.bias1 += np.random.normal(0, stdev, self.bias1.shape)
		self.bias2 += np.random.normal(0, stdev, self.bias2.shape)
		self.bias3 += np.random.normal(0, stdev, self.bias3.shape)

	def save_weights(self, filename):
		np.savez(filename, FC1=self.FC1, 
							FC2=self.FC2, 
							FC3=self.FC3, 
							bias1=self.bias1, 
							bias2=self.bias2, 
							bias3=self.bias3)

	def load_weights(self, filename):
		npzfile = np.load(filename)
		self.FC1 = npzfile['FC1']
		self.FC2 = npzfile['FC2']
		self.FC3 = npzfile['FC3']
		self.bias1 = npzfile['bias1']
		self.bias2 = npzfile['bias2']
		self.bias3 = npzfile['bias3']


def run_simulation(network, env, count=5, view=False):
	scores = []
	for _ in range(count):
		observation = env.reset()
		if view:
			env.render()
		score = 0
		for i in range(LIFETIME):
			score -= 1
			action = network.predict(observation)[0,0]
			observation, reward, done, info = env.step(action)
			if reward == 0:
				break
		env.close()
		scores.append(score)
	return np.mean(scores)

def evolve(filename='networks/best_network.npz', show=False, generations=20):
	env = gym.make(ENV_NAME)
	networks = []

	for i in range(100):
		networks.append(NN())

	for generation in range(generations):
		scores = []
		for index, network in enumerate(networks):
			print("Generation {:02d}, network {:03d}".format(generation, index), end='\r')
			score = run_simulation(network, env)
			scores.append(score)
		networks = [networks[x] for x in np.argsort(scores)[::-1]]
		scores = np.sort(scores)[::-1]

		if show:
			demonstrate(networks[0])

		new_networks = []
		for i in range(10):
			for j in range(10):
				new_network = copy.deepcopy(networks[i])
				if j > 0:
					new_network.mutate()
				new_networks.append(new_network)
		networks = new_networks
		print("Generation {}: Best Score={}".format(generation, scores[0]))



	best_network = networks[0]
	if filename:
		best_network.save_weights(filename)
	return best_network


def demonstrate(network):
	env = gym.make(ENV_NAME)
	observation = env.reset()
	env.render()
	score = 0
	for i in range(LIFETIME):
		score -= 1
		action = network.predict(observation)[0,0]
		observation, reward, done, info = env.step(action)
		env.render()
		if reward == 0:
			break
	env.close()
	return score

if __name__ == '__main__':
	if len(sys.argv[1:]) > 0:
		network = NN()
		network.load_weights(sys.argv[1])
	else:
		network = evolve(show=False, generations=100)
	while True:
		score = demonstrate(network)
		print("Ended after {} steps".format(-score))

