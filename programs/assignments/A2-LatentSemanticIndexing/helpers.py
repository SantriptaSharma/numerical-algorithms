from collections import Counter
import csv

import plotly.express as px
import numpy as np
import nltk

tokenizer = nltk.tokenize.TreebankWordTokenizer()
lemmatizer = nltk.WordNetLemmatizer()
stop_words = ['the', 'of', 'and', 'in', 'a', 'to', 'an', 'on', 'from', 'by', 'for', 'with']

def get_term_freqs(text):
	global tokenizer, lemmatizer, stop_words

	
	toks = [lemmatizer.lemmatize(tok).lower() for tok in tokenizer.tokenize(text) if tok.isalpha()]
	toks = [tok for tok in toks if tok not in stop_words]

	return Counter(toks)

def embed(text: str, global_term_freqs, normalize=False):
	global_term_indices = {term: i for i, term in enumerate(global_term_freqs.keys())}

	term_freqs = get_term_freqs(text)

	doc = np.zeros(len(global_term_freqs))

	for term, freq in term_freqs.items():
		if term not in global_term_freqs:
			continue
		
		if normalize:
			freq /= global_term_freqs[term]

		doc[global_term_indices[term]] = freq
	
	return doc

def doc_query(embedding, svd):
	K = svd["K"]

	U = svd["U"]
	S = svd["S"]

	# multiplying by S^{-1} == dividing by S (S is a diagonal matrix (scaling mat) represented as a vector)
	return np.dot(embedding, U[:, :K]) / S[:K]

def get_doc_indices_by_distance(query, svd):
	K = svd["K"]

	Vh = svd["Vh"]
	
	distances = np.dot(query, Vh[:K, :])
	return np.argsort(distances)[::-1]

def extract_categories(ddn, start=0, end=3) -> list:
	num = int(ddn.split(".")[0])

	categories = set()

	for i in range(start, end):
		snap_to = 10**i
		snapped = (num // snap_to) * snap_to
		categories.add(f"{snapped:03}")

	return sorted(list(categories))

def rank_drop_indices(S, order=1.1):
	shifted_indices = np.arange(1, len(S))

	singular_guys_shifted = np.r_[S[shifted_indices], [1]]
	changes = (S / singular_guys_shifted)[:-1]

	return np.where(changes > 10**order)[0]

def truncate_svd(NP_SVD, K):
	Uk = np.copy(NP_SVD.U)
	Uk[:, K:] = 0

	Sk = np.copy(NP_SVD.S)
	Sk[K:] = 0

	Vhk = np.copy(NP_SVD.Vh)
	Vhk[K:, :] = 0

	return {"U": Uk, "S": Sk, "Vh": Vhk, "K": K}