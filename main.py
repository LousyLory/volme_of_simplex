"""
Coded by Archan while at UMass Amherst
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lg
import sys
import argparse
import copy

def construct_simplex(dimensions, range_):
	"""
	inputs:
	dimensions: int, dimension of space
	range_: list, 2 vals, min max range of the vertices' coordinates
	output:
	vertices: 2D array, size-[dimnesions+1, n]

	The construction should follow the construction of the paper:
	'https://arxiv.org/pdf/1905.06208.pdf'
	"""
	vertices = np.zeros((dimensions+1, dimensions))
	for i in range(dimensions+1):
		if i < dimensions:
			vertices[i, :] = np.zeros_like(vertices[i, :])
			vertices[i, i] = 1
			if range_[1] is not 1:
				vertices[i,:] = range_[1]*vertices[i,:]
			if range_[0] is not 0:
				vertices[i, vertices[i, :]==0] = range_[0]
		else:
			vertices[i, :] = np.ones_like(vertices[i, :])
			vertices[i, :] = range_[0]*vertices[i, :]
	return vertices

def sample_rand_vec(a, b, V):
	"""
	a and b are ints
	V is a set of n vectors (rows) of b dimensions
	"""
	vec = np.random.randn(a, b)
	dot_prod = np.dot(vec, V.T)
	return vec, dot_prod

def draw_a(V):
	"""
	input:
	V: list of vertices of a simplex, 2D numpy array
	output:
	a: a vector in unit sphere S^(n-1) such that a^T.v is not equal to
	   a^T.w for any distinct vecrtices (v,w) \in V^2
	This is inspired from Lasserre's works (2.2) of the paper
	'Volume of slices and sections of the simplex in closed form'
	"""
	vec, dot_prod = sample_rand_vec(1, V.shape[-1], V)
	while len(np.unique(dot_prod)) < V.shape[-1]:
		vec, dot_prod = sample_rand_vec(1, V.shape[-1], V)
	return vec

def compute_volume_section(V, a, t):
	"""
	inputs:
	V: list of vertices of the simplex, 2D numpy array
	a: vector from the unit ball of dimension V.shape[-1], numpy array
	output:
	volume: volume of the section formed with V and a
	"""
	volume = 0
	"""
	# costly way
	for i in range(len(V)):
		V_dash = copy.deepcopy(V)
		# actual computation
		numerator = t-np.dot(a, V[i,:])
		# relu
		numerator = max(0, numerator)
		# power to relu
		numerator = np.power(numerator, len(V))

		# subtraction (v-w) for all w and a given v
		V_dash = V_dash-V_dash[i,:]
		# product a^T*(v-w)
		prod_vec = np.dot(a, V_dash.T)
		# drop v == v
		prod_vec = prod_vec[prod_vec != 0]
		# product over all w in V and w not equal to v
		prod_val = np.prod(prod_vec)

		# the final fraction in this loop
		fraction = numerator/prod_val
		
		volume = volume+fraction
	"""
	# less costly way
	# compute a^T*v for all v in V
	prod_vec = np.dot(a, V.T)
	for i in range(len(V)):
		sample_prod = copy.deepcopy(prod_vec)
		
		sample_prod = sample_prod[]
	return volume

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dimensions", default=2, type=int, help="the R^n space in which the simplex exists")
	parser.add_Argument("--boundary_val", default=0.3, type=float, help="a^T*x <= t")
	args = parser.parse_args()

	Simplex = construct_simplex(args.dimensions, [0, 1])
	A = draw_a(Simplex)
	vol = compute_volume_section(Simplex, A, args.boundary_val)
	print(Simplex, A)

if __name__ == "__main__":
	main()
