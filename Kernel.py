import Signal
import numpy as np

class Kernel():
    def __init__(self,fun):
        self.__function = fun
        
    def evaluate(self,sig1,sig2):
        if sig1.get_size() == sig2.get_size():
            return self.__function(sig1.get_signal(),sig2.get_signal())
        return 0
    
    
def polynomial_kernel(c,d):
    ker = Kernel(lambda x,y : ( np.dot(x,y) + c )**d)
    return ker
    
def gaussian_kernel(c):
    ker = Kernel( lambda x,y : np.exp(np.linalg.norm(x-y)**2/c) )
    return ker