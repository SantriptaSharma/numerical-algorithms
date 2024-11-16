import numpy as np

def get_random_symmetric_matrix(n: int, a = 0.0, b = 1.0) -> np.ndarray:
	""" generates a random symmetric matrix of uniform values in [a, b) """
	
	A = np.random.uniform(a, b, (n, n))
	return np.tril(A) + np.tril(A, -1).T

def make_upper_hessenberg(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""
	perform (n - 2) size householders to produce an upper hessenberg matrix with the same eigencharacteristics as A
	assumes symmetric A

	returns H
	"""

	_, n = A.shape

	A = A.copy()
	Q = np.eye(n)

	if n == 1:
		return A, Q

	e = np.zeros(n - 1)
	e[0] = 1

	for j in range(n - 2):
		x = A[j + 1:, j]

		x = np.sign(x[0]) * np.linalg.norm(x) * e[:n-j-1] + x
		x = x / np.linalg.norm(x)

		# perform A <- (I - 2vv^t) A (pre mult)
		A[j+1:, j:] -= 2 * np.outer(x, np.dot(x, A[j+1:, j:]))

		# perform A <- A (I - 2vv^t)^t = A (I - 2v^tv) (post-mult, similarity transform)
		A[j:, j+1:] -= 2 * np.outer(np.dot(A[j:, j+1:], x), x)

		Q[j+1:, :] -= 2 * np.outer(x, np.dot(x, Q[j+1:, :]))
		
	return A, Q.T

def hessenberg_qr(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""
	performs 2x2 householders to zero out the subdiagonal of a symmetric H, producing an upper triangle R and orthog Q st H = QR

	returns Q, R
	"""

	_, n = H.shape

	R = H.copy()
	Qt = np.eye(n)

	if n == 1:
		return Qt, R


	e = np.array([1, 0])

	for j in range(n - 1):
		x = R[j:j+2, j]
		x = np.sign(x[0]) * np.linalg.norm(x) * e + x
		x = x / np.linalg.norm(x)

		R[j:j+2, j:] -= 2 * np.outer(x, np.dot(x, R[j:j+2, j:]))
		Qt[j:j+2, :] -= 2 * np.outer(x, np.dot(x, Qt[j:j+2, :]))

	return Qt.T, R