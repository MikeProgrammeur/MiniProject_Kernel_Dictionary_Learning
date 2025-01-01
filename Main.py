import Signal as sg
import Kernel as kn
import KernelDictionaryLearning as kdl
import numpy as np
import matplotlib.pyplot as plt

polynomial_kernel = kn.polynomial_kernel(c=0,d=2)
signals_r1 = sg.synthetic_circle(radius=1,data_number=1500)
signals_r2 = sg.synthetic_circle(radius=2,data_number=1500)
trained_kdl_r1 = kdl.KernelDictionaryLearning(signals=signals_r1,kernel=polynomial_kernel,sparsity_level=3,atom_number=30,n_iter=80)
trained_kdl_r1.learn()
trained_kdl_r2 = kdl.KernelDictionaryLearning(signals=signals_r2,kernel=polynomial_kernel,sparsity_level=3,atom_number=30,n_iter=80)
test = trained_kdl_r2.learn()