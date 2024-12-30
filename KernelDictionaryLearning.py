import numpy as np

class KernelDictionaryLearning():
    def __init__(self,signals,kernel,sparsity_level,n):
        self.__signals = signals # list of yi
        self.__sparsity_level = sparsity_level #T0
        self.__n = n #number of signals
        self.__kernel = kernel
        
        
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
            vec[i] = self.__kernel.evaluate(z,self.signals.get_signal(i))
        return vec
        
    def KOMP(self, z): # a verifier
        "Kernel orthogonal matching pursuit for input object signal z"
        s = 0
        I = set()  # Ensemble des indices
        x = np.zeros(self.__n)  # Vecteur des coefficients
        z_hat = np.zeros(self.__n)  # Approximation initiale de z

        KYY = self.gen_KYY()  # Matrice KYY
        KzY = self.gen_KzY(z)  # Vecteur KzY

        while s < self.__sparsity_level:
            # Calcul de τ_i
            tau = np.zeros(self.__n)
            for i in range(self.__n):
                if i not in I:
                    tau[i] = KzY[i] - np.dot(z_hat.T, KYY[i, :])

            # Trouver l'indice maximal de |τ_i|
            imax = np.argmax(np.abs(tau))
            I.add(imax)

            # Mettre à jour x
            A_I = KYY[list(I), :][:, list(I)]
            KzY_I = KzY[list(I)]
            x[list(I)] = np.linalg.inv(A_I).dot(KzY_I)

            # Mettre à jour l'approximation de z
            z_hat = KYY.dot(x)

            s += 1

        return x  # Retourne le vecteur x