class gd_2d_plain:

    def __init__(self,function,gradient):
        
        self.function = function
        self.gradient = gradient
        
    def minimize(self, x0, eta, max_iter=100000, tol=1e-5):
        import numpy as np       
        
        x = x0
        fval = []       # path of function values
        x_path = []     # path of x vector
        x_path.append(x)
        fval.append(self.function(x))
        grad = self.gradient(x)
        
        y = self.function(x)
        n_iter = 0
        
        while np.linalg.norm(grad) >= tol:
            grad = self.gradient(x)         # updating gradient
            x = x - eta * grad                # updating x
            x_path.append(x)                # recording new x in the x_path
            y = self.function(x)            # updating function value with new x
            fval.append(y)                  # recording updated function value in the fval
            n_iter = n_iter + 1
            if n_iter >= max_iter:
                break
            else:
                continue
        
        self.grad = grad
        self.norm_grad= np.linalg.norm(grad)
        self.fval_path = np.array(fval)
        self.fval = y
        self.x_path = np.array(x_path)
        self.x = x
        self.niter = n_iter
        if np.linalg.norm(grad) <= tol:
            print("Optimization terminated successfully")
        elif (np.linalg.norm(grad) > tol) and np.array(fval).min()==y:
            raise ValueError("Convergence failed : Insufficient iteration")
        else:
            raise ValueError("Convergence failed")
        
        return self
    
class gd_2d_momentum:

    def __init__(self,function,gradient):
        
        self.function = function
        self.gradient = gradient
        
    def minimize(self, x0, eta, alpha=0.9, max_iter=100000, tol=1e-8):
        import numpy as np       
        
        x = np.array(x0)
        v = [np.array([0,0])]         # initial step
        fval = []       # path of function values
        x_path = []     # path of x vector
        x_path.append(x)
        fval.append(self.function(x))
        grad = self.gradient(x)
        
        y = self.function(x)
        n_iter = 0
        
        while np.linalg.norm(grad) >= tol:
            grad = self.gradient(x)             # updating gradient         
            v.append(alpha*v[-1] + eta*grad)    # updating v
            x = x - v[-1]                       # updating x
            x_path.append(x)                    # recording new x in the x_path
            y = self.function(x)                # updating function value with new x
            fval.append(y)                      # recording updated function value in the fval
            n_iter = n_iter + 1
            if n_iter >= max_iter:
                break
            else:
                continue
        
        self.grad = grad
        self.norm_grad= np.linalg.norm(grad)
        self.fval_path = np.array(fval)
        self.fval = y
        self.x_path = np.array(x_path)
        self.x = x
        self.niter = n_iter
        if np.linalg.norm(grad) <= tol:
            print("Optimization terminated successfully")
        elif (np.linalg.norm(grad) > tol) and np.array(fval).min()==y:
            raise ValueError("Convergence failed : Insufficient iteration")
        else:
            raise ValueError("Convergence failed")
        
        return self

class gd_2d_NAG:

    def __init__(self,function,gradient):
        
        self.function = function
        self.gradient = gradient
        
    def minimize(self, x0, eta, alpha=0.9, mu=0.5, max_iter=100000, tol=1e-8):
        import numpy as np       
        
        x = np.array(x0)
        v = [np.array([0,0])]         # initial step
        fval = []       # path of function values
        x_path = []     # path of x vector
        x_path.append(x)
        fval.append(self.function(x))
        grad = self.gradient(x)
        
        y = self.function(x)
        n_iter = 0
        
        while np.linalg.norm(grad) >= tol:
            grad = self.gradient(x - mu*v[-1])    # updating gradient         
            v.append(alpha*v[-1] + eta*grad)    # updating v
            x = x - v[-1]                       # updating x
            x_path.append(x)                    # recording new x in the x_path
            y = self.function(x)                # updating function value with new x
            fval.append(y)                      # recording updated function value in the fval
            n_iter = n_iter + 1
            if n_iter >= max_iter:
                break
            else:
                continue
        
        self.grad = grad
        self.norm_grad= np.linalg.norm(grad)
        self.fval_path = np.array(fval)
        self.fval = y
        self.x_path = np.array(x_path)
        self.x = x
        self.niter = n_iter
        if np.linalg.norm(grad) <= tol:
            print("Optimization terminated successfully")
        elif (np.linalg.norm(grad) > tol) and np.array(fval).min()==y:
            raise ValueError("Convergence failed : Insufficient iteration")
        else:
            raise ValueError("Convergence failed")
        
        return self
