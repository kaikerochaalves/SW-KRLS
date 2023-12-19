# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np

class SW_KRLS:
    def __init__(self, N = 30, sigma = 0.5, c = 0.1):
        #self.hyperparameters = pd.DataFrame({})
        self.parameters = pd.DataFrame(columns = ['Kinv', 'alpha', 'm', 'Dict', 'yn'])
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        # Hyperparameters and parameters
        # Maximum size of the dictionary
        self.N = N
        # Sigma
        self.sigma = sigma
        # Regularization constant
        self.c = c
         
    def fit(self, X, y):

        # Compute the number of samples
        n = X.shape[0]
        
        # Initialize the first input-output pair
        x0 = X[0,].reshape(-1,1)
        y0 = y[0]
        
        # Initialize the consequent parameters
        self.Initialize_SW_KRLS(x0, y0)

        for k in range(1, n):

            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
                      
            # Update the consequent parameters
            kn1 = self.SW_KRLS(x, y[k])
            
            # Compute the output
            Output = self.parameters.loc[0, 'alpha'].T @ kn1
            
            # Store the results
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output )
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(y[k]) - Output )
        return self.OutputTrainingPhase
            
    def predict(self, X):

        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T

            # Compute k
            kn1 = np.array(())
            for ni in range(self.parameters.loc[0, 'Dict'].shape[1]):
                kn1 = np.append(kn1, [self.Kernel(self.parameters.loc[0, 'Dict'][:,ni].reshape(-1,1), x)])
            kn1 = kn1.reshape(-1,1)
            # Compute the output
            Output = self.parameters.loc[0, 'alpha'].T @ kn1
            # Storing the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output )
               
        return self.OutputTestPhase

    def Kernel(self, x1, x2):
        k = np.exp( - ( 1/2 ) * ( (np.linalg.norm( x1 - x2 ))**2 ) / ( self.sigma**2 ) )
        return k
    
    def Initialize_SW_KRLS(self, x, y):
        Kinv = np.eye(1) / (1 + self.c)
        alpha = Kinv * y
        yn = np.eye(1) * y
        NewRow = pd.DataFrame([[Kinv, alpha, 1., x, yn]], columns = ['Kinv', 'alpha', 'm', 'Dict', 'yn'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
        # Initialize first output and residual
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def SW_KRLS(self, x, y):
        i = 0
        # Update dictionary
        self.parameters.at[i, 'Dict'] = np.hstack([self.parameters.loc[i, 'Dict'], x])
        # Update yn
        self.parameters.at[i, 'yn'] = np.vstack([self.parameters.loc[i, 'yn'], y])
        # Compute k
        k = np.array(())
        for ni in range(self.parameters.loc[i, 'Dict'].shape[1]):
            k = np.append(k, [self.Kernel(self.parameters.loc[i, 'Dict'][:,ni].reshape(-1,1), x)])
        kn1 = k[:-1].reshape(-1,1)
        knn = self.Kernel(x, x)
        # Compute Dinv
        D_inv = self.parameters.loc[i, 'Kinv']
        # Update Kinv
        g = 1 / ( ( knn + self.c ) - kn1.T @ D_inv @ kn1 )
        f = ( - D_inv @ kn1 * g ).flatten()
        E = D_inv - D_inv @ kn1 @ f.reshape(1,-1)
        sizeKinv = self.parameters.loc[i, 'Kinv'].shape[0] - 1
        self.parameters.at[i, 'Kinv'] = E 
        self.parameters.at[i, 'Kinv'] = np.lib.pad(self.parameters.loc[i, 'Kinv'], ((0,1),(0,1)), 'constant', constant_values=(0))
        sizeKinv = self.parameters.loc[i,  'Kinv'].shape[0] - 1
        self.parameters.at[i, 'Kinv'][sizeKinv,sizeKinv] = g
        self.parameters.at[i, 'Kinv'][0:sizeKinv,sizeKinv] = f
        self.parameters.at[i, 'Kinv'][sizeKinv,0:sizeKinv] = f
        
        # Verify if the size of the dictionary is greater than N
        if self.parameters.loc[i, 'Dict'].shape[1] > self.N:
            # Remove the oldest element in the dictionary
            self.parameters.at[i, 'Dict'] = np.delete(self.parameters.loc[i, 'Dict'], 0, 1)
            # Update yn
            self.parameters.at[i, 'yn'] = np.delete(self.parameters.loc[i, 'yn'], 0, 0)
            # Update k
            k = np.delete(k, 0, 0)
            # Compute Dinv
            G = self.parameters.loc[i, 'Kinv'][1:,1:]
            f = self.parameters.loc[i, 'Kinv'][1:,0].reshape(-1,1)
            ft = self.parameters.loc[i, 'Kinv'][0,1:].reshape(1,-1)
            e = self.parameters.loc[i, 'Kinv'][0,0]
            D_inv = G - ( f @ ft ) / e
            # Update Kinv
            self.parameters.at[i, 'Kinv'] = D_inv

        # Compute alpha
        self.parameters.at[i, 'alpha'] = self.parameters.loc[i,  'Kinv'] @ self.parameters.loc[i, 'yn']
        return k.reshape(-1,1)