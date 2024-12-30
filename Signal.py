class Signal():
    def __init__(self,signal,size):
        self.__signal = signal
        self.__size = size
        
    def get_signal(self):
        return self.__signal
    
    def get_size(self):
        return self.__size
    
class Signals():
    def __init__(self,signals):
        self.__signals = signals
        
    def get_signal(self,i):
        return self.__signals[i]
        
    