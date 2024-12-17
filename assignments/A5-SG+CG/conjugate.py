import numpy as np

EPS = 1e-12

def conj_grad(A: np.ndarray, b: np.ndarray, x0: np.ndarray | None = None, max_iterations = 500) -> tuple[np.ndarray, float, int, list[np.ndarray], list[np.ndarray]]:
	"""
	uses the method of conjugate gradients to minimise the homogeneous (c = 0) quadratic form given by (A, b)

	returns the minimising vector x, minimal value of the function, number of iterations till convergence, values of the vector x at each iteration, and residuals at each iteration
	"""

	if not (A.shape[0] == A.shape[1] and np.all(A.T == A) and np.all(np.linalg.eigvals(A) > 0)):
		raise Exception("A must be symmetric & positive definite")
	
	n = A.shape[0]

	b = b.squeeze()

	if not (b.ndim == 1 and b.shape[0] == n):
		raise Exception(f"b must be a vector of size {n}")
	
	if x0 is None:
		x = np.random.uniform(-5, 5, n)
	else:
		x = x0.squeeze()

		if not (x.ndim == 1 and x.shape[0] == n):
			raise Exception(f"x0 must be a vector of size {n}")
	
	# analytical form of the gradient of a quadratic form
	calc_grad = lambda x: A @ x - b

	xs = [x]

	residual = -calc_grad(x)
	residuals = [residual]

	d = residual

	for k in range(max_iterations):
		res_sq = np.dot(residual, residual)

		alpha = res_sq / np.dot(d, A @ d)

		x = x + alpha * d
		xs.append(x)

		# again, could be sped up w the update rule, but dont care for now
		new_res = -calc_grad(x)

		beta = np.dot(new_res, new_res) / res_sq

		residual = new_res
		residuals.append(residual)
    
		if np.linalg.norm(residual) < EPS:
			break

		d = residual + beta * d


	return x, np.dot(x, 1/2 * A @ x - b), k + 1, xs, residuals

if __name__ == "__main__":
	A = np.array([[3, 2], [2, 6]])
	b = np.array([2, -8])

	x0 = np.array([500, 230])

	x, val, its, xs, residuals = conj_grad(A, b, x0)

	print(f"found minima at {x} in {its} iterations: {val}")