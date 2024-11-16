import numpy as np
from eigen import practical_qr

def svd(A: np.ndarray) -> tuple[np.ndarray, np.array, np.ndarray]:
	m, n = A.shape
	transpose = m < n

	if transpose:
		A = A.T
		m, n = n, m

	corr = A.T @ A

	evals, evecs = practical_qr(corr)
	evals = np.abs(evals)

	sorted_idx = np.flip(np.argsort(evals))
	evals = evals[sorted_idx]
	evecs = evecs[:, sorted_idx]

	S = np.sqrt(evals)
	V = evecs
	U = np.zeros((m, n))

	for i in range(n):
		U[:, i] = A @ V[:, i] / S[i]
	
	if transpose:
		return V, S, U.T
	
	return U, S, V.T
	