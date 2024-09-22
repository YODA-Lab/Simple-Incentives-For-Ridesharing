"""
numpy version of the fairness functions from DECAF.
"""
import numpy as np

class FairnessFunction:
	"""
	Abstract class for fairness functions that operate on a vector of non-negative values.
	"""
	def __init__(self, name: str):
		self.name = name

	def evaluate(self, Z: np.ndarray)->float:
		"""
		Evaluate the fairness function on a vector of non-negative values.
		"""
		raise NotImplementedError
	
	def is_maximizing(self)->bool:
		"""
		Returns whether the fairness function is maximizing or minimizing.
		"""
		raise NotImplementedError
	
	def get_metric(self, Z: np.ndarray)->float:
		"""
		Alias for evaluate, adding a sign to indicate whether the fairness function is maximizing or minimizing.
		"""
		return self.evaluate(Z) * (-1)**(self.is_maximizing()+1)
	
	def get_bounds(self)->tuple:
		"""
		Returns the bounds of the fairness function.
		"""
		raise NotImplementedError
	
	def __str__(self)->str:
		return self.name
	
	def __repr__(self)->str:
		return self.name
	
	def compute_gradient(self, Z: np.ndarray)->np.ndarray:
		"""
		Compute the gradient of the fairness function with respect to all elements of Z, evaluated at Z.
		"""
		raise NotImplementedError

class JAXFairnessFunction(FairnessFunction):
	"""
	Decorator for fairness functions that use JAX.
	"""
	def __init__(self, name: str):
		super().__init__(name)
		# gradient_fn = jax.grad(self.get_metric)
		# self.compute_gradient = jax.jit(gradient_fn)
	
	def evaluate(self, Z: np.ndarray)->float:
		raise NotImplementedError
	
	def grad_based_fairness_rewards(self, Z1: np.ndarray, Z2: np.ndarray)->np.ndarray:
		"""
		Compute the fairness rewards for each agent.
		Z1: vector of values for the first time step.
		Z2: vector of values for the second time step.
		Assume decomposition does not exist, use the default implementation.
		"""
		raise NotImplementedError

	def get_fairness_rewards(self, Z1: np.ndarray, Z2: np.ndarray)->np.ndarray:
		"""
		Compute the fairness rewards for each agent.
		Z1: vector of values for the first time step.
		Z2: vector of values for the second time step.
		"""
		return self.grad_based_fairness_rewards(Z1, Z2)
	
	def get_naive_fairness_rewards(self, Z1: np.ndarray, Z2: np.ndarray)->np.ndarray:
		"""
		Equally divide the fairness rewards between the agents.
		"""
		DelF = self.get_metric(Z2) - self.get_metric(Z1)
		n_agents = Z1.shape[0]
		return np.ones(n_agents) * DelF / n_agents

class Variance(JAXFairnessFunction):
	"""
	Variance fairness function.
	"""
	def __init__(self):
		super().__init__("Variance")
	
	def evaluate(self, Z: np.ndarray)->float:
		#jax for automatic differentiation
		return np.var(Z)
	
	def is_maximizing(self)->bool:
		return False
	
	def get_bounds(self)->tuple:
		return (0, np.inf)
	
	def get_fairness_rewards(self, Z1: np.ndarray, Z2: np.ndarray)->np.ndarray:
		#Manual decomposition
		assert Z1.shape == Z2.shape
		n_agents = Z1.shape[0]
		zbar = np.mean(Z1)
		z2bar = np.mean(Z2)
		scores = -((Z2 - z2bar)**2 - (Z1 - zbar)**2)/n_agents
		return scores

class Gini(JAXFairnessFunction):
	"""
	Gini index fairness function.
	"""
	def __init__(self, epsilon: float = 1e-6):
		super().__init__("Gini")
		self.epsilon = epsilon

	def evaluate(self, Z: np.ndarray)->float:
		#jax for automatic differentiation
		# Check edge case: if all values are 0, add epsilon to avoid division by 0
		#Create a copy of z to avoid modifying the original
		Z = np.array(Z) + self.epsilon

		n = Z.shape[0]
		MAD = np.sum(np.abs(Z[:, None] - Z))
		MAD /= 2 * n**2 * np.mean(Z)
		return MAD
	
	def is_maximizing(self)->bool:
		return False
	
	def get_bounds(self)->tuple:
		return (0, 1)
	
	def get_fairness_rewards(self, Z1: np.ndarray, Z2: np.ndarray)->np.ndarray:
		#Manual decomposition
		assert Z1.shape == Z2.shape
		n_agents = Z1.shape[0]
		zbar = np.mean(Z1)
		z2bar = np.mean(Z2)
		if zbar == 0:
			zbar = self.epsilon
		if z2bar == 0:
			z2bar = self.epsilon
		z1_term = np.sum(np.abs(Z1[:, None] - Z1) / (2 * n_agents**2 * zbar), axis=1)
		z2_term = np.sum(np.abs(Z2[:, None] - Z2) / (2 * n_agents**2 * z2bar), axis=1)
		scores = z1_term - z2_term
		return scores

class Maximin(JAXFairnessFunction):
	"""
	Maximin fairness function.
	"""
	def __init__(self):
		super().__init__("Maximin")
	
	def evaluate(self, Z: np.ndarray)->float:
		#jax for automatic differentiation
		return np.min(Z)
	
	def is_maximizing(self)->bool:
		return True
	
	def get_bounds(self)->tuple:
		return (0, np.inf)
	
	def get_fairness_rewards(self, Z1: np.ndarray, Z2: np.ndarray)->np.ndarray:
		#Manual decomposition
		assert Z1.shape == Z2.shape

		min_1 = np.min(Z1)
		min_2 = np.min(Z2)
		scores = np.where(Z1 == min_1, Z2 - Z1, 0)
		scores += np.where(Z2 == min_2, Z2 - Z1, 0)
		scores += (min_2 - min_1) / Z1.shape[0] #Shaping reward to give a global signal
		
		#normalize to have same sum as min_2 - min_1
		if np.sum(scores) != 0:
			scores = scores * (min_2 - min_1) / np.sum(scores)
		
		return scores


class JainIndex(JAXFairnessFunction):
	"""
	Jain index fairness function.
	"""
	def __init__(self, epsilon: float = 1e-6):
		super().__init__("Jain Index")
		self.epsilon = epsilon
	
	def evaluate(self, Z: np.ndarray)->float:
		#jax for automatic differentiation
		# Epsilon if largest value is <eps, else 0 addition
		trigger = -np.min(np.array([np.max(Z) - self.epsilon, 0]))
		Z = np.array(Z) + self.epsilon * trigger
		
		n = Z.shape[0]
		z_sum = np.sum(Z)
		z2_sum = np.sum(Z**2)
		return z_sum**2 / (n * z2_sum)
	
	def is_maximizing(self)->bool:
		return True
	
	def get_bounds(self)->tuple:
		return (0, 1)
	
	# This decomposition is not correct, the correct decomposition is not implemented yet
	# def get_fairness_rewards(self, Z1: np.ndarray, Z2: np.ndarray)->np.ndarray:
	# 	#Manual decomposition
	# 	assert Z1.shape == Z2.shape
	# 	n_agents = Z1.shape[0]
	# 	z_sum = np.sum(Z1)
	# 	z2_sum = max([np.sum(Z1**2), self.epsilon])
	# 	z_sum2 = np.sum(Z2)
	# 	z2_sum2 = max(np.sum(Z2**2), self.epsilon)
	# 	term1 = 2*z_sum/(n_agents*z2_sum)
	# 	term2 = -2*Z1*(z_sum**2)/(n_agents*(z2_sum**2))
	# 	term1_2 = 2*z_sum2/(n_agents*z2_sum2)
	# 	term2_2 = -2*Z2*(z_sum2**2)/(n_agents*(z2_sum2**2))
	# 	scores = term1_2 - term1 + term2_2 - term2
	# 	return scores

class AlphaFair(JAXFairnessFunction):
	"""
	Alpha fairness function.
	"""
	def __init__(self, alpha: float=1, epsilon: float = 1e-6):
		super().__init__(f"Alpha Fairness (alpha={alpha})")
		self.alpha = alpha
		self.epsilon = epsilon
	
	def evaluate(self, Z: np.ndarray)->float:
		#jax for automatic differentiation

		#add epsilon to avoid division by 0, and to limit the effect of 0 values
		Z = np.array(Z) + self.epsilon
		
		if self.alpha == 1:
			return np.sum(np.log(Z))
		return np.sum(Z**(1-self.alpha))/(1-self.alpha)
	
	def is_maximizing(self)->bool:
		return True
	
	def get_bounds(self)->tuple:
		return (-np.inf, np.inf)
	
	def get_fairness_rewards(self, Z1: np.ndarray, Z2: np.ndarray)->np.ndarray:
		#Manual decomposition
		assert Z1.shape == Z2.shape
		n_agents = Z1.shape[0]
		scores = np.zeros(n_agents)
		if self.alpha == 1:
			scores = np.log(Z2) - np.log(Z1)
		else:
			scores = ((Z2)**(1-self.alpha) - (Z1)**(1-self.alpha))/(1-self.alpha)
		return scores


class GGF(JAXFairnessFunction):
	"""
	Generalized Gini fairness function.
	"""
	def __init__(self, weights=None, epsilon: float = 1e-6):
		super().__init__("GGF")
		#Check the weights are strictly decreasing
		if weights is not None:
			assert np.all(np.diff(weights) < 0)
		self.weights = weights
		self.epsilon = epsilon
	
	def set_weights(self, n_agents: int):
		# Create a default weight vector, reducing sequence. Normalize to sum to 1
		# self.weights = np.array([1/(i+1) for i in range(n_agents)])
		self.weights = np.array([1/(2**i) for i in range(n_agents)])
		# self.weights /= np.sum(self.weights)
		assert np.all(np.diff(self.weights) < 0)

	def evaluate(self, Z: np.ndarray)->float:
		if self.weights is None:
			self.set_weights(Z.shape[0])
		sorted_indices = np.argsort(Z)
		sorted_ = Z[sorted_indices]
		GGF = np.sum(self.weights * sorted_)
		return GGF
	
	def is_maximizing(self)->bool:
		return True
	
	def get_bounds(self)->tuple:
		return (0, np.inf)
	
	def get_fairness_rewards(self, Z1: np.ndarray, Z2: np.ndarray)->np.ndarray:
		#Manual decomposition
		"""
		Idea: Match the weights to the sorted values. Let w_i be the weight of z_i after sorting.
		Then, R_f(i) = w_i'*z_i' - w_i*z_i, where z_i' is the value of z_i in the second time step.
		TODO: This is not well aligned. It gives negative weights for agents that overtake other agents.
		TODO: But maybe that is desired?
		"""
		assert Z1.shape == Z2.shape
		if self.weights is None:
			self.set_weights(Z1.shape[0])
		sorted_Z1 = np.argsort(Z1)
		sorted_Z2 = np.argsort(Z2)

		ordered_weights = self.weights[sorted_Z1]
		ordered_weights2 = self.weights[sorted_Z2]
		# scores = ordered_weights2 * Z2 - ordered_weights * Z1
		# print("Ordered Weights:", ordered_weights2)
		scores = ordered_weights2 * (Z2 - Z1) # Trying with just post ranking
		# scores = ordered_weights2 * rewards * 100# Trying with just rewards
		return scores
	

def fairness_router(fairness_function: str, **kwargs)->FairnessFunction:
	"""
	Router to select the fairness function.
	"""
	func_map = {
		"variance": Variance,
		"gini": Gini,
		"maximin": Maximin,
		"jain_index": JainIndex,
		"alpha_fair": AlphaFair,
		"ggf": GGF
	}
	if fairness_function in func_map:
		retf = func_map[fairness_function](**kwargs)
		# print(f"Using fairness function: {retf}")
		return retf
	else:
		raise ValueError(f"Fairness function {fairness_function} not supported.")