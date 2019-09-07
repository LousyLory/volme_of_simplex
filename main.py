"""
Coded by Archan while at UMass Amherst
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lg
import sys
import argparse
import copy
from scipy.special import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
	a^T.w for any distinct vecrtices (v,w) in V^2
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
	# set up volume
	volume = 0
	# less costly way is to compute prod_vec only once
	# compute a^T*v for all v in V
	prod_vec = np.dot(a, V.T)
	
	# for loop for the sum computations
	for i in range(len(V)):
		# copy the prod_Vec for further processing
		sample_prod = copy.deepcopy(prod_vec)
		# take out the vertex in consideration
		sample_prod = sample_prod[sample_prod != sample_prod[0,i]]
		# subtract the val from the remaining values
		sample_prod = prod_vec[0,i] - sample_prod
		# compute the product
		denominator = np.prod(sample_prod)
		# numerator
		numerator = t - prod_vec [0,i]
		# fraction
		fraction = numerator/denominator
		# volume
		volume = volume+fraction

	# multiply factorial n
	volume = volume/factorial(len(V))
	return volume

def visualize(Simplex, A):
	"""
	visualize function to verify the values obtained
	"""
	if Simplex.shape[-1] > 3:
		print("can't visualize more than 3 dimensions")
		sys.exit(1)
	else:
		pass
	# visualize the simplex
	if Simplex.shape[-1] > 2:
		fig = plt.figure().gca(projection='3d')
		fig.plot_surface(Simplex[:,0], Simplex[:,1], Simplex[:,2], alpha=0.2)
		ax = plt.gca(projection='3d')
		ax.hold(True)
		fig.plot_surface(A[:,0], A[:,1], A[:,2], alpha=0.2)
		plt.savefig("visualization/3D.png")
	else:
		pass
	return None

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dimensions", default=2, type=int, help="the R^n space in which the simplex exists")
	parser.add_argument("--boundary_val", default=0.3, type=float, help="a^T*x <= t")
	parser.add_argument("--visualization", default=False, type=bool, help="choose whether to visualize or not")
	args = parser.parse_args()

	Simplex = construct_simplex(args.dimensions, [0, 1])
	A = draw_a(Simplex)
	vol = compute_volume_section(Simplex, A, args.boundary_val)
	if args.visualization:
		visualize(Simplex, A)
	print(Simplex, A, vol)

if __name__ == "__main__":
	main()
