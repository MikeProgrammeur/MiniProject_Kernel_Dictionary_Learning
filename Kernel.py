import Signal
class Kernel():
    def __init__(self,fun,dim):
        self.__function = fun
        self.__dim = dim
        
    def evaluate(self,sig1,sig2):
        temp = sig1.get_size() #stockage pour éviter de calculer deux fois
        if self.__dim == temp and temp == sig2.get_size():
            return self.__function(sig1.get_signal(),sig2.get_signal())
        return 0
    