import Signal as sg
import Kernel as kn
import KernelDictionaryLearning as kdl
import numpy as np
import matplotlib.pyplot as plt

data_sample_number = 200
n_iter = 10

polynomial_kernel = kn.polynomial_kernel(c=0,d=2)
signals_r1 = sg.synthetic_circle(radius=1,data_number=data_sample_number)
signals_r2 = sg.synthetic_circle(radius=2,data_number=data_sample_number)
trained_kdl_r1 = kdl.KernelDictionaryLearning(signals=signals_r1,kernel=polynomial_kernel,sparsity_level=3,atom_number=30,n_iter=n_iter)
trained_kdl_r1.learn()
trained_kdl_r2 = kdl.KernelDictionaryLearning(signals=signals_r2,kernel=polynomial_kernel,sparsity_level=3,atom_number=30,n_iter=n_iter)
test = trained_kdl_r2.learn()

resolution_plot = 100
coord_lim = 2
x = np.linspace(-coord_lim, coord_lim, resolution_plot)
y = np.linspace(-coord_lim, coord_lim, resolution_plot)

Z = np.array([ np.array([ trained_kdl_r1.KOMP(sg.Signal([xi,yi],2))[1]/trained_kdl_r2.KOMP(sg.Signal([xi,yi],2))[1] for xi in x]) for yi in y ])

plt.contourf(x,y, Z, levels=50, cmap="coolwarm")
plt.colorbar(label="Error Ratio")
plt.title("Error Ratio Map")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()



# i think the current error is in ksvd, when we encouter an atom that isn't used what must we do ? Skip the procedure? but that 
# is not specified in the pseudo code. But i think we skip svd for k-th atom if we notice that w_k is empty, that is why there is an error
# with empy delta variable.
# Init of A must also be verified 
# Check why the eigvalues of EkrT KYY EkR are complex while this matrix is supposed to be symmetric