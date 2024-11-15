import numpy as np
from utils import make_upper_hessenberg, hessenberg_qr

ESSENTIALLY_ZERO = 10e-10

def practical_qr_no_smart_qr(A: np.ndarray) -> tuple[np.array, np.ndarray]:
	""" 
	runs the practical qr algorithm (with RQ shifts) on the given matrix, returning its eigenvalues 
	doesn't use the upper hessenberg fact (dumb qr)
	"""

	def qr_iteration(A: np.ndarray):
		_, n = A.shape
		
		QQ = np.eye(n)

		if n == 1:
			return A, QQ

		for k in range(5000):
			for j in range(n - 1):
				if np.abs(A[j, j+1]) < ESSENTIALLY_ZERO or np.abs(A[j+1, j]) < ESSENTIALLY_ZERO:
					A[j, j + 1] = 0
					A[j + 1, j] = 0

					evals_a1, evecs_a1 = qr_iteration(A[:j+1, :j+1])
					evals_a2, evecs_a2 = qr_iteration(A[j+1:, j+1:])

					A[:j+1, :j+1] = evals_a1
					A[j+1:, j+1:] = evals_a2

					Q = np.zeros((n, n))
					Q[:j+1, :j+1] = evecs_a1
					Q[j+1:, j+1:] = evecs_a2

					QQ = QQ @ Q

					return A, QQ

			u = A[n - 1, n - 1] * np.eye(n)
			Q, R = np.linalg.qr(A - u)
			A = (R @ Q) + u

			QQ = QQ @ Q

		return A, QQ
	
	H, Qh = make_upper_hessenberg(A)
	evals, evecs = qr_iteration(H)

	evecs = Qh @ evecs
	return evals.diagonal(), evecs


def practical_qr(A: np.ndarray) -> tuple[np.array, np.ndarray]:
	""" 
	runs the practical qr algorithm (with RQ shifts) on the given matrix, returning its eigenvalues 
	"""

	def qr_iteration(A: np.ndarray):
		_, n = A.shape
		
		QQ = np.eye(n)

		if n == 1:
			return A, QQ

		for k in range(5000):
			for j in range(n - 1):
				if np.abs(A[j, j+1]) < ESSENTIALLY_ZERO or np.abs(A[j+1, j]) < ESSENTIALLY_ZERO:
					A[j, j + 1] = 0
					A[j + 1, j] = 0

					evals_a1, evecs_a1 = qr_iteration(A[:j+1, :j+1])
					evals_a2, evecs_a2 = qr_iteration(A[j+1:, j+1:])

					A[:j+1, :j+1] = evals_a1
					A[j+1:, j+1:] = evals_a2

					Q = np.zeros((n, n))
					Q[:j+1, :j+1] = evecs_a1
					Q[j+1:, j+1:] = evecs_a2

					QQ = QQ @ Q

					return A, QQ

			u = A[n - 1, n - 1] * np.eye(n)
			Q, R = hessenberg_qr(A - u)
			A = (R @ Q) + u

			QQ = QQ @ Q

		return A, QQ
	
	H, Qh = make_upper_hessenberg(A)
	evals, evecs = qr_iteration(H)

	evecs = Qh @ evecs
	return evals.diagonal(), evecs