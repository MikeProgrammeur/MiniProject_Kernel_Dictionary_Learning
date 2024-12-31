import numpy as np

class KernelDictionaryLearning():
    def __init__(self,signals,kernel,sparsity_level,atom_number,dim,n_iter):
        self.__signals = signals # list of yi
        self.__sparsity_level = sparsity_level #T0
        self.__n = self.__signals.get_sig_number() #number of signals
        self.__kernel = kernel
        self.__dim = dim
        self.__atom_number = atom_number #K
        self.__matrix_A = np.zeros((self.__n,self.__atom_number))
        self.__matrix_X = np.zeros((self.__n,self.__n))
        self.__KYY = np.zeros((self.__n,self.__n))
        self.__n_iter = n_iter
        
    def gen_KYY(self):
        mat = np.zeros((self.__n,self.__n))
        for i in range(self.__n):
            for j in range(i+1):
                mat[i,j] = self.__kernel.evaluate(self.signals.get_signal(i),self.signals.get_signal(j))
                if i!=j:
                    mat[j,i] = mat[i,j]
        return mat
                    
    def gen_KzY(self,z):
        vec = np.zeros(self.__n)
        for i in range(self.__n):
            vec[i] = self.__kernel.evaluate(z,self.signals.get_signal_i(i))
        return np.expand_dims(vec,axis=1).T
        
    def KOMP(self, z): # a verifier
        "Kernel orthogonal matching pursuit for input object signal z"
        # Variables initialization
        s = 0
        I = set()  # Ensemble des indices
        x = np.zeros(self.__atom_number)  # Vecteur des coefficients
        z_hat = np.expand_dims(np.zeros(self.__n),axis=1).T # Approximation initiale de z vecteur colonne
        print(f"{z_hat.shape} should be ({self.__n},1)")
        KzY = self.gen_KzY(z)  # Vecteur KzY
        print(f"{KzY.shape} should be (1,{self.__n})")
        print(f"{self.__matrix_A[:,0].size} should be ({self.__n},1)")

        while s < self.__sparsity_level: # REPEAT (1) - (6) T_o times
            # (1) COMPUTE TAU_I's
            tau = np.zeros(self.__atom_number)
            for i in range(self.__atom_number):
                if i not in I:
                    tau[i] = (KzY - np.dot(z_hat.T, self.__KYY))@self.__matrix_A[:,i]

            # (2) FIND BEST RESIDUAL APPROXIMATION
            imax = np.argmax(np.abs(tau))
            
            # (3) UPDATE THE INDEX SET 
            I.add(imax)

            # (4) COMPUTE NEW ORTHOGONAL PROJECTION
            A_I = self.__matrix_A[:,list(I)]
            x = np.linalg.inv(A_I.T@self.__KYY@A_I)@(KzY@A_I).T

            # (5) UPDATE Z APPROXIMATION
            z_hat = A_I.dot(x)

            # (6) INCREMENT s
            s += 1
            
        # Output x_out in RK s.t. x_out(I(j))=x(j) for all j in I
        x_out = np.zeros(self.__atom_number)
        for j,Ij in enumerate(list(I)):
            x_out[Ij] = x[j]
        return x_out 
    
    def KSVD(self,X):
        " dictionary update step given X a sparse coding of signals"
        # X shape is (K,n)
        for k in range(self.__atom_number): # repeat (1) - (5) for each column of A
            # (1) FIND SIGNALS USING A_k FOR REPRESENTATION
            w_k = set()
            for i in range(self.__n): # They say in range K (and also N) in article but i think it is n
                if X[k,i] != 0 :
                    w_k.add(i)
            omega_k = np.zeros((self.__n,len(w_k)))
            for i,w_k_i in enumerate(w_k):
                omega_k[w_k_i,i] = 1
                
            # (2)  DEFINE THE REPRESENTATION ERROR MATRIX
            sum = np.zeros((self.__n,self.__n))
            for j in range(self.__atom_number):
                if j!=k:
                    sum+=np.dot(self.__matrix_A[:,j],X[j])
            E_k = np.eye(self.__n) - sum
            
            # (3) RESTRICT E_k
            EkR = E_k@omega_k
            
            # (4) APPLY SVD DECOMPOSITION
            delta,V = np.linalg.eig(EkR.T@self.__KYY@EkR)
            largest_delta_index = np.argmax(delta)
            largest_delta = delta[largest_delta_index]
            v1 = V[:,largest_delta_index]
            a_k = 1/np.sqrt(largest_delta) * EkR @ v1
            
            # (5) UPDATE k-th COLUMN OF A
            self.__matrix_A[:,k] = a_k
            
        
    
    def learn(self):
        self.__KYY = self.gen_KYY() 
        
        for i in range(self.__n_iter):
            for j in range(self.__n):
                self.__matrix_X[:,j] = self.KOMP(self.__signals.get_signal_i(j))
            self.KSVD(self.__matrix_X)
        