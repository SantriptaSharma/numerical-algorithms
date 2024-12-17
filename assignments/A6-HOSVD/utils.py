import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
import os

def create_dyntex_tensor(dir):
	pics = os.listdir(dir)

	T = len(pics)

	ordered = sorted(pics, key=lambda x: int(x.split('.')[0]))

	# Load the first image to get the shape
	first = plt.imread(os.path.join(dir, ordered[0]))
	H, W = first.shape

	tensor = np.zeros((H, W, T))

	for i, pic in enumerate(ordered):
		tensor[:, :, i] = plt.imread(os.path.join(dir, pic))

	return tensor

def visualise_dyntex(tensor, start = 0, count = 5):
	SIZE_PER_IMG = 10
	plt.ioff()
	fig, axs = plt.subplots(1, count, figsize=(SIZE_PER_IMG * count, SIZE_PER_IMG))

	for i in range(count):
		axs[i].imshow(tensor[:, :, start + i], cmap="gray")
		axs[i].axis('off')
	
	fig.tight_layout()
	plt.ion()
	return fig

def get_temporal_mean(tensor):
	return np.mean(tensor, axis=2, keepdims=True)

def sample(G):
	return G @ np.random.normal(0, 1, G.shape[0])

def reconstruct(hosvd, x, M):
	Z = tl.tenalg.mode_dot(hosvd.core, hosvd.factors[0], mode=0)
	Z = tl.tenalg.mode_dot(Z, hosvd.factors[1], mode=1)
	Z = tl.tenalg.mode_dot(Z, x, mode=2)

	return Z + M[:, :, 0]