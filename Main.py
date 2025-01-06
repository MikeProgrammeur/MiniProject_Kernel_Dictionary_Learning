# IMPORT ZONE
import Signal as sg
import Kernel as kn
import KernelDictionaryLearning as kdl
import numpy as np
import matplotlib.pyplot as plt
import random

# HYPERPARAMETERS OF THE PROBLEM
data_sample_number = 400
n_iter = 10
sparsity_level=3
atom_number=30

# VARIABLES OF THE PROBLEM
polynomial_kernel = kn.polynomial_kernel(c=1,d=2)
gaussian_kernel = kn.gaussian_kernel(c=10)
signals_r1 = sg.synthetic_circle(radius=1,data_number=data_sample_number)
signals_r2 = sg.synthetic_circle(radius=2,data_number=data_sample_number)
trained_kdl_r1 = kdl.KernelDictionaryLearning(signals=signals_r1,kernel=polynomial_kernel,sparsity_level=sparsity_level,atom_number=atom_number,n_iter=n_iter)
trained_kdl_r1.learn()
trained_kdl_r2 = kdl.KernelDictionaryLearning(signals=signals_r2,kernel=polynomial_kernel,sparsity_level=sparsity_level,atom_number=atom_number,n_iter=n_iter)
trained_kdl_r2.learn()

# CONSTRUCTION OF THE IMAGE TO PLOT
resolution_plot = 100
coord_lim = 3
x = np.linspace(-coord_lim, coord_lim, resolution_plot)
y = np.linspace(-coord_lim, coord_lim, resolution_plot)
epsilon = 1e-20 # regularizaion parameter to avoid division by zero
Z = np.array([ np.array([ np.abs(trained_kdl_r1.KOMP(sg.Signal([xi,yi],2))[1])/(np.abs(trained_kdl_r2.KOMP(sg.Signal([xi,yi],2))[1])+epsilon) for xi in x]) for yi in y ])
#Z = np.array([ np.array([ 1 if np.abs(trained_kdl_r1.KOMP(sg.Signal(np.array([xi,yi]),2))[1])>np.abs(trained_kdl_r2.KOMP(sg.Signal(np.array([xi,yi]),2))[1]) else -1 for xi in x]) for yi in y ])
#Z = np.array([ np.array([ np.abs(trained_kdl_r1.KOMP(sg.Signal([xi,yi],2))[1]) for xi in x]) for yi in y ])
log_ratios = np.log10(Z)

# DISPLAY THE IMAGE
plt.imshow(log_ratios, cmap='coolwarm', origin='lower', extent=[-coord_lim, coord_lim, -coord_lim, coord_lim])
plt.colorbar(label='Logarithmic Error Ratio (log10)')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Error Ratio: Kernel K-SVD')
plt.show()

