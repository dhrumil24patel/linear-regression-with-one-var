import numpy as np

class Learner:

	def __init__(self, learning_rate = 0.01, iterations = 1000):
		self.learning_rate = learning_rate
		self.iterations = iterations

		self.theta0 = 0
		self.theta1 = 0

	def learn(self, X, y):
		print("learning...")
		m = len(X)

		for iteration in range(0, self.iterations):
			hypothysis = self.hypothysis(X)

			temp0 = self.theta0 - self.learning_rate * (1/m) * (np.sum(np.multiply((hypothysis - y), X)))
			temp1 = self.theta1 - self.learning_rate * (1/m) * (np.sum(hypothysis - y))

			self.theta0 = temp0
			self.theta1 = temp1

			print("iteration ",iteration," cost ",self.cost(X,y))

		print("theta0 ",self.theta0)
		print("theta1 ",self.theta1)

	# 1/2m(sum(y`-y)**2)
	def cost(self, X, y):
		m = len(X)
		cost = (1/(2*m)*np.sum((self.hypothysis(X) - y)**2))
		return cost

	# y = theta0 * X + theta1
	def hypothysis(self, X):
		y = np.add(np.multiply(X, self.theta0), self.theta1)
		return y

	def test(self, x):
		print("test for x ",x," output y: ",self.hypothysis(x))

def demo():
	print("somthing")
	rng = range(0,1000)
	X = np.zeros(len(rng), dtype = float)
	y = np.zeros(len(rng), dtype = float)

	for i in rng:
		X[i] = i/100;
		y[i] = (2*i)/100

	learner = Learner(0.05, 500)
	learner.learn(X,y)

	learner.test(120)
	learner.test(7.8)
	learner.test(90)

if __name__ == '__main__':
	demo()