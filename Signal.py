import numpy as np

class Signal():
    def __init__(self,signal,size):
        self.__signal = signal # numpy array
        self.__size = size # int
        
    def get_signal(self):
        return self.__signal
    
    def get_size(self):
        return self.__size
    
class Signals():
    def __init__(self,signals):
        self.__signals = signals # python list of Signal object
        self.__signal_number = len(self.__signals)
        
    def get_signal_i(self,i):
        " return the i-th signal"
        if i<self.__signal_number:
            return self.__signals[i]
        return 0
    
    def get_sig_number(self):
        return self.__signal_number
    
def synthetic_circle(radius,data_number):
    " return a object Signals containing 'data_number' 2D points from circle of radius 'radius' "
    signals = []
    for i in range(data_number):
        o = np.pi * np.random.uniform(0, 2)
        s = Signals( radius*np.array([ np.cos(o),np.sin(o)] ), 2)
        signals.append(s)
    
    signals_object = Signals(signals)