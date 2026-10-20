import numpy as np

class perceptron:
    def __init__(self, learning_rate=0.01, mx_epochs=1000):
        self.lr = learning_rate
        self.mx_epochs = mx_epochs
    
    def signum(self, v):
        if v >= 0:
            return 1
        else:
            return -1
        
    def perc(self,X,Y):
        n_samples, n_feats = X.shape
        
        self.w = np.random.uniform(-0.01, 0.01, n_feats)
        self.b = 0.0
        
        self.errors_per_epoch = []
        
        for epoch in range(self.mx_epochs):
            errors = 0
            
            for i in range(n_samples):
                x=X[i]
                y=Y[i]  
                
                v=np.dot(self.w, x) + self.b
                y_pred=self.signum(v)
                error = y - y_pred
                
                if error != 0:
                    self.w += self.lr * error * x
                    self.b += self.lr * error
                    errors += 1
                
                self.errors_per_epoch.append(errors)
                
                if errors == 0:
                    break
        return self 
                
    def predict(self, X):
        y_pred = []
        for x in X:
            v = np.dot(self.w, x) + self.b
            y_pred.append(self.signum(v))
        return np.array(y_pred)
        
        
         
         