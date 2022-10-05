import utils as u
import numpy as np

from scipy.optimize import minimize
from constants import *

# Grab the dataset
classA, classB, inputs, targets, N = u.get_dataset()

# Select the kernel
our_kernel = u.poly_kernel

# Pre-calculate the P matrix
P = u.pre_compute_P(N, targets, inputs, our_kernel)

# Helper functions used for the minimize function
def objective(a):
    return (1/2) * np.dot(a, np.dot(a, P)) - np.sum(a)

def zerofun(a):
    return np.dot(a, targets)


# Obtain the resulting dictionary from the minimize function
start = np.zeros(N)
bounds = u.bound(N)
constraints= {'type':'eq', 'fun':zerofun} 
ret = minimize(objective, start, bounds=bounds, constraints=constraints)
alpha = ret['x']

# Extract the non-zero values
non_zeros = u.extract_non_zeros(alpha, inputs, targets, N)

# Plot the results
u.plot(classA, classB, non_zeros, our_kernel)