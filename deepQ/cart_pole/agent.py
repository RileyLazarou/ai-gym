import numpy as np
from collections import deque
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size		# input size
		self.action_size = action_size		# output size
		self.lr = 1e-3						# learning rate
		self.gamma = 0.95					# discount rate
		self.epsilon = 1e0					# exploration/exploitation
		self.epsilon_min = 1e-2				# minimum epsilon
		self.epsilon_decay = 0.995			# epsilon decay rate
		self.model = self._build_model()	# Q "function"
		self.memory = deque(maxlen=1000)	# memory for replaying 

	def _build_model(self):
		input_layer = layers.Input((self.state_size,))
		X = layers.Dense(32, activation='relu')(input_layer)
		X = layers.Dense(16, activation='relu')(X)
		X = layers.concatenate([input_layer, X])
		X = layers.Dense(16, activation='relu')(X)
		X = layers.Dense(16, activation='relu')(X)
		X = layers.concatenate([input_layer, X])
		output_layer = layers.Dense(self.action_size, activation='linear')(X)
		model = Model(input_layer, output_layer)
		model.compile(optimizer=Adam(lr=self.lr), loss='mse')
		return model

	def remember(self, state, action, reward, next_state, done):
		# Store state, action, reward, next_state, done
		state = np.array(state).reshape((1, -1))
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state, explore=True):
		if explore and np.random.rand() < self.epsilon:
			return np.random.randint(self.action_size)
		state = np.array(state).reshape((1,-1))
		estimated_rewards = self.model.predict(state)
		return np.argmax(estimated_rewards[0])

	def predict_scores(self, state):
		state = np.array(state).reshape((1,-1))
		estimated_rewards = self.model.predict(state)
		return estimated_rewards

	def replay(self, batch_size):
		minibatch = [self.memory[x] for x in np.random.randint(0, len(self.memory), batch_size)]
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + self.gamma * np.max(self.predict_scores(next_state))
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


	def save_model(self, filename):
		self.model.save(filename)

	def load_model(self, filename):
		self.model = load_model(filename)

