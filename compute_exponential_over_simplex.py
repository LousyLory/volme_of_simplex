import numpy as np
import copy

def vol_with_poles(c, len_c):
	"""
	Computes volume of an exponential over a linear form using Lasserre's form
	"""
	volume = 0
	col_repeat_matrix = np.repeat(c.T, len_c, axis=1)
	#row_repeat_matrix = np.repeat(c, len_c, axis=0)
	row_repeat_matrix = col_repeat_matrix.T
	difference_matrix = col_repeat_matrix - row_repeat_matrix
	product_list = np.zeros_like(c)
	for i in range(len_c):
		product_list = 
	"""
	for i in range(len(c)):
		dinominator = 1
	"""
	return None

def main():
	len_c = 5
	c = np.random.randn(len_c,1)
	vol_with_poles(c.T, len_c)

if __name__ == "__main__":
	main()


