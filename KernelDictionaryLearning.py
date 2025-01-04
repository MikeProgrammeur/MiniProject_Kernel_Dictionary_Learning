import numpy as np
import random as rd
import Kernel as kn
import Signal as sg

class KernelDictionaryLearning():
    def __init__(self, signals : sg.Signals, kernel : kn.Kernel, sparsity_level : int, atom_number : int, n_iter : int):
        self.__signals = signals # list of yi
        self.__sparsity_level = sparsity_level #T0
        self.__n = self.__signals.get_sig_number() #number of signals : n in the article
        self.__kernel = kernel
        self.__atom_number = atom_number # : K in the article
        self.__matrix_A = np.zeros((self.__n,self.__atom_number))
        self.__matrix_X = np.zeros((self.__atom_number,self.__n))
        self.__KYY = np.zeros((self.__n,self.__n))
        self.__n_iter = n_iter
        
    def gen_KYY(self):
        " generate the matrix KYY where KYYij = <K(Yi)|K(Yj)> "
        mat = np.zeros((self.__n,self.__n))
        for i in range(self.__n):
            for j in range(i+1):
                mat[i,j] = self.__kernel.evaluate(self.__signals.get_signal_i(i),self.__signals.get_signal_i(j))
                if i!=j:
                    mat[j,i] = mat[i,j]
        #print(np.allclose(mat,mat.T))
        return mat
                    
    def gen_KzY(self, z : sg.Signal):
        " generate the 1-row matrix KzY where KzYi = <z|K(Yi)> "
        vec = np.zeros(self.__n)
        for i in range(self.__n):
            vec[i] = self.__kernel.evaluate(z,self.__signals.get_signal_i(i))
        return np.expand_dims(vec,axis=1).T
        
    def KOMP(self, z: sg.Signal): # a verifier
        "Kernel orthogonal matching pursuit for input object signal z, return x the atom activation vector and the norm of the last residual"
        # Variables initialization
        s = 0
        I = set()  # Ensemble des indices
        x = np.zeros(self.__atom_number)  # Vecteur des coefficients
        z_hat = np.expand_dims(np.zeros(self.__n),axis=1) # Approximation initiale de z vecteur colonne
        #print(f"z_hat shape is {z_hat.shape} and should be ({self.__n},1)")
        KzY = self.gen_KzY(z)  # Vecteur KzY
        #print(f"KzY shape is {KzY.shape} and should be (1,{self.__n})")
        #print(f"a_i shape is {np.expand_dims(self.__matrix_A[:,0],axis=1).shape} and should be ({self.__n},1)")

        while s < self.__sparsity_level: # REPEAT (1) - (6) T_0 times
            # (1) COMPUTE TAU_I's
            tau = np.zeros(self.__atom_number)
            for i in range(self.__atom_number):
                if i not in I:
                    tau[i] = (KzY - np.dot(z_hat.T, self.__KYY))@np.expand_dims(self.__matrix_A[:,i],axis=1)

            # (2) FIND BEST RESIDUAL APPROXIMATION
            imax = np.argmax(np.abs(tau))
            #print(tau)
            if np.isclose(tau[imax],0) and s>0: # don't need to look for more atom is representation is exact
                break
            
            # (3) UPDATE THE INDEX SET 
            I.add(imax)

            # (4) COMPUTE NEW ORTHOGONAL PROJECTION
            A_I = self.__matrix_A[:,list(I)]
            #print(f"A_I shape is {A_I.shape} and should be ({self.__n},{len(I)})")
            x = np.linalg.inv(A_I.T@self.__KYY@A_I)@(KzY@A_I).T

            # (5) UPDATE Z APPROXIMATION
            z_hat = A_I.dot(x) 

            # (6) INCREMENT s
            s += 1
            
        # Output x_out in RK s.t. x_out(I(j))=x(j) for all j in I
        x_out = np.zeros(self.__atom_number)
        for j,Ij in enumerate(list(I)):
            x_out[Ij] = x[j]
            
        norm_residual_error = np.squeeze(self.__kernel.evaluate(z,z) + z_hat.T@self.__KYY@z_hat-2*KzY@z_hat)
        
        return x_out,norm_residual_error
    
    def KSVD(self,X):
        " dictionary update step given X a sparse coding of signals"
        # X shape is (K,n)
        for k in range(self.__atom_number): # repeat (1) - (5) for each column of A
            # (1) FIND SIGNALS USING A_k FOR REPRESENTATION
            w_k = set()
            for i in range(self.__n): # They say in range K (and also N) in article but i think it is n
                if not(np.isclose(X[k,i],0)) :
                    w_k.add(i)
            if len(w_k) != 0 :
                omega_k = np.zeros((self.__n,len(w_k)))
                for i,w_k_i in enumerate(w_k):
                    omega_k[w_k_i,i] = 1
                    
                # (2)  DEFINE THE REPRESENTATION ERROR MATRIX
                sum = np.zeros((self.__n,self.__n))
                for j in range(self.__atom_number):
                    if j!=k:
                        #print(np.linalg.norm(np.outer(self.__matrix_A[:,j],X[j])))
                        sum+=np.outer(self.__matrix_A[:,j],X[j])
                E_k = np.eye(self.__n) - sum
                
                # (3) RESTRICT E_k
                EkR = E_k@omega_k
                
                # (4) APPLY SVD DECOMPOSITION
                #print(EkR.T@self.__KYY@EkR,EkR,omega_k)
                #aaaa = EkR.T@self.__KYY@EkR
                #print(np.allclose(aaaa,aaaa.T))
                #delta,V = np.linalg.eigh(EkR.T@self.__KYY@EkR)
                V,delta,VT = np.linalg.svd(a = EkR.T@self.__KYY@EkR, hermitian = True)
                #print(delta)
                largest_delta_index = np.argmax(delta)
                largest_delta = delta[largest_delta_index]
                v1 = V[:,largest_delta_index]
                a_k = 1/np.sqrt(largest_delta) * EkR @ v1
                
                # (5) UPDATE k-th COLUMN OF A
                self.__matrix_A[:,k] = a_k
            else : 
                self.__matrix_A[:,k] = np.random.randn(self.__n)
            
    def init_X(self):
        " initialize representation matrix X which is sparse"
        index_up_to_K = [k for k in range(self.__atom_number)]
        for i in range(self.__n): # Set T0 random elements of each column in X to be 1
            atom_activated_index = rd.sample(index_up_to_K, self.__sparsity_level)
            for j in atom_activated_index:
                self.__matrix_X[j,i] = 1
        #print(self.__matrix_X)
    
    def init_A(self):
        " initialize A the atom representation dictionary"
        if self.__n >= self.__atom_number :
            index_up_to_n = [n for n in range(self.__n)]
            atom_dictionary_index = rd.sample(index_up_to_n, self.__sparsity_level)
            for k,atom in enumerate(atom_dictionary_index):
                self.__matrix_A[atom,k]=1
        #self.__matrix_A = np.random.randn(self.__n,self.__atom_number # maybe better to initialize A like this??
                
    def calc_objective_fun(self):
        " return the objective function || phi(Y) - phi(Y)AX||²"
        temp = np.eye(self.__n) - self.__matrix_A@self.__matrix_X
        return np.trace(temp.T@self.__KYY@temp)
    
    def learn(self):
        " learn a dictionary and a representation matrix over phi(Y) "
        # Compute kernel image
        self.__KYY = self.gen_KYY()
        
        # Initialize with random sparse coding at the beginning
        self.init_X()
        self.init_A()
        
        
        for i in range(self.__n_iter):
            # ALTERNATE n_iter TIMES sparse coding : (1) and dictionary update : (2)
            for j in range(self.__n):
                self.__matrix_X[:,j],_ = self.KOMP(self.__signals.get_signal_i(j))
            #print(self.__matrix_X)
            self.KSVD(self.__matrix_X)
            print(f"Total representation error is {self.calc_objective_fun()} at step {i}")
        return "successfull"
        