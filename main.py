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

def newline(p1, p2):
	ax = plt.gca()
	xmin, xmax = ax.get_xbound()

	if(p2[0] == p1[0]):
		xmin = xmax = p1[0]
		ymin, ymax = ax.get_ybound()
	else:
		ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
		ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

		l = mlines.Line2D([xmin,xmax], [ymin,ymax])
		ax.add_line(l)
	return l


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
	vertices[0, :] = np.ones_like(vertices[0,:])
	for i in range(1, dimensions+1):
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

def sample_rand_vec(a, b, V, t):
	"""
	a and b are range of values to draw values from
	V is a set of n vectors (rows) of b dimensions
	t is the radius of a hypersphere
	"""
	vec = np.random.uniform(low=a, high=b, size=(1, V.shape[-1]))
	while np.linalg.norm(vec) == 0:
		vec = np.random.uniform(low=a, high=b, size=(1, V.shape[-1]))
	# transform to the surface on the unit sphere
	if np.linalg.norm(vec) != 0:
		norm_vec = vec/np.linalg.norm(vec)
		vec = t*norm_vec

	dot_prod = np.dot(vec, V.T)
	return vec, dot_prod

def draw_a(V,t):
	"""
	input:
	V: list of vertices of a simplex, 2D numpy array
	t: a^T*x <= t for all x in R^n
	output:
	a: a vector in unit sphere S^(n-1) such that a^T.v is not equal to
	a^T.w for any distinct vecrtices (v,w) in V^2
	This is inspired from Lasserre's works (2.2) of the paper
	'Volume of slices and sections of the simplex in closed form'
	"""
	vec, dot_prod = sample_rand_vec(0, 1, V, t)
	while len(np.unique(dot_prod)) < V.shape[-1]:
		vec, dot_prod = sample_rand_vec(0, 1, V)
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
		#print("sample_prod",sample_prod)
		# compute the product
		denominator = np.prod(sample_prod)
		#print("denominator", denominator)
		# numerator
		numerator_relu = max(0,t - prod_vec [0,i])
		#print("numerator_relu",numerator_relu)
		numerator = np.power(numerator_relu, V.shape[-1])
		#print("numerator", numerator)
		# fraction
		fraction = numerator/denominator
		#print("fraction",fraction)
		# volume
		volume = volume+fraction

	# multiply factorial n
	#print(V.shape[-1])
	volume = volume/factorial(V.shape[-1])
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
	if Simplex.shape[-1] == 2:
		for i in range(len(Simplex)):
			# pick the point you want to treat as root
			pt_null = Simplex[i,:]
			new_range = range(len(Simplex))
			new_range.remove(i)
			for j in new_range:
				target_pt = Simplex[j, :]
				plt.plot([pt_null[0], target_pt[0]], [pt_null[1], target_pt[1]])
		# plot the line
		xs = [A[0,0],0]
		ys = [0, A[0,1]]
		plt.plot(xs, ys)
		plt.savefig('visualization/2D.png')
	
	if Simplex.shape[-1] > 2:
		raise NotImplementedError
	
		plt.savefig('visualization/3D.png')		
	pass
	return None

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dimensions", default=2, type=int, help="the R^n space in which the simplex exists")
	parser.add_argument("-b", "--boundary_val", default=0.3, type=float, help="a^T*x <= t")
	parser.add_argument("-v", "--visualization", default=False, type=bool, help="choose whether to visualize or not")
	args = parser.parse_args()

	Simplex = construct_simplex(args.dimensions, [0, 1])
	A = draw_a(Simplex, args.boundary_val)
	#A = np.array(([[0.02467475, 0.99969553]]))
	vol = compute_volume_section(Simplex, A, args.boundary_val)
	print(args.visualization)
	if args.visualization:
		visualize(Simplex, A)
	print(Simplex, A, np.linalg.norm(A), vol)

if __name__ == "__main__":
	main()
