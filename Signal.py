import numpy as np
import matplotlib.pyplot as plt

class Signal():
    def __init__(self,signal,size):
        self.__signal = signal # numpy array of shape (size,)
        self.__size = size # int
        
    def get_signal(self):
        "return the signal as a numpy array"
        return self.__signal
    
    def get_size(self):
        "return size/dimension of the signal"
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
        "return the number of signals"
        return self.__signal_number
    
def synthetic_circle(radius,data_number):
    " return a object Signals containing 'data_number' 2D points from circle of radius 'radius' "
    signals_list = []
    for _ in range(data_number):
        o = np.pi * np.random.uniform(0, 2)
        s = Signal( signal = radius*np.array([ np.cos(o),np.sin(o)] ), size = 2)
        signals_list.append(s)
    
    signals_object = Signals(signals=signals_list)
    return signals_object

def display_signal(signals1 : Signals, signals2: Signals, title : str, class1name : str, class2name : str):
    "plot each point of a class in a specific color"
    x_coords_1 = []
    y_coords_1 = []
    x_coords_2 = []
    y_coords_2 = []
    
    for i in range(signals1.get_sig_number()):
        signal = signals1.get_signal_i(i)
        signal_array = signal.get_signal()
        x_coords_1.append(signal_array[0])
        y_coords_1.append(signal_array[1])
    for i in range(signals2.get_sig_number()):
        signal = signals2.get_signal_i(i)
        signal_array = signal.get_signal()
        x_coords_2.append(signal_array[0])
        y_coords_2.append(signal_array[1])
    
    # Plot the points
    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords_1, y_coords_1, color='blue', alpha=0.6, label=class1name)
    plt.scatter(x_coords_2, y_coords_2, color='red', alpha=0.6, label=class2name)
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()