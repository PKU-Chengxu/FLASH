import numpy as np 


class StructuredUpdate:
    def __init__(self, k, seed=0):
        self.seed = seed
        self.k = k
    
    def struc_update(self, grads):
        # if needed
        np.random.seed(self.seed)
        grads_new = []
        shape_old = []

        for grad in grads:
            shape = grad.shape
            shape_old.append(shape)
            if len(shape) != 2:
                grad = grad.reshape([int(np.prod(shape[:-1])), shape[-1]])
            
            a = np.random.random([grad.shape[0], self.k])
            a_inv = np.linalg.pinv(a)
            b = np.dot(a_inv, grad)
            grads_new.append(b)

        return self.seed, shape_old, grads_new
    
    def regain_grad(self, shape_old, grad_new):
        # if needed
        np.random.seed(self.seed)

        grads = []
        for i in range(len(shape_old)):
            shape = shape_old[i]
            grad = grad_new[i]
            a = np.random.random([int(np.prod(shape[:-1])), self.k])
            g = np.dot(a, grad).reshape(shape)
            grads.append(g)
        return grads
