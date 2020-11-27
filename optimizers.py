import numpy as np

class Optimizer:
    def __init__(self, learning_rate=None, name=None):
        self.learning_rate = learning_rate
        self.name = name

    def config(self, layers):
        # sets up empty cache dictionaries 
        pass

    def optimize(self, idx, layers: list, grads: dict, *args):
        '''# Args: Takes in idx of the layer, list of the layers and the gradients as a dictionary 
            Performs updates in the list of layers passed into it'''
        pass 


class SGDM(Optimizer):
    '''  Momentum builds up velocity in any direction that has consistent gradient'''

    def __init__(self, learning_rate=1e-2, mu_init=0.5, max_mu=0.99, demon=False, beta_init=0.9, **kwargs):
            super().__init__(**kwargs)
            self.mu_init = mu_init
            self.max_mu = max_mu
            self.demon = demon
            if self.demon:
                self.beta_init = beta_init
            self.m = dict()

    def config(self, layers):
        for i in layers.keys():
            self.m[f'W{i}'] = 0
            self.m[f'b{i}'] = 0

    def optimize(self, idx, layers, grads, epoch_num, steps):
        # increase mu by a factor of 1.2 every epoch until max_mu is reached (only applicable for momentum and nesterov momentum)
        mu = min(self.mu_init * 1.2 ** (epoch_num - 1), self.max_mu)

        if self.demon:
            p_t = 1 - epoch_num / self.epochs 
            mu = self.beta_init * p_t / ((1 - self.beta_init) + self.beta_init * p_t) 

        self.m[f'W{idx}'] = self.m[f'W{idx}'] * mu - self.learning_rate * grads[f'dW{idx}']
        self.m[f'b{idx}'] = self.m[f'b{idx}'] * mu - self.learning_rate * grads[f'db{idx}']

        layers[idx].W += self.m[f'W{idx}']
        layers[idx].b += self.m[f'b{idx}']



class Nesterov(SGDM):
    '''Nesterov's Accelerated Momentum: https://arxiv.org/pdf/1212.0901v2.pdf'''
    def __init__(self, learning_rate, **kwargs):
        self.learning_rate = learning_rate
        super().__init__(**kwargs)


    def optimize(self, idx, layers, grads, epoch_num, steps):
        # increase mu by a factor of 1.2 every epoch until max_mu is reached (only applicable for momentum and nesterov momentum)
        mu = min(self.mu_init * 1.2 ** (epoch_num - 1), self.max_mu)
        if self.demon:
            p_t = 1 - epoch_num / self.epochs 
            mu = self.beta_init * p_t / ((1 - self.beta_init) + self.beta_init * p_t) 

        mW_prev =  np.array(self.m[f'W{idx}'], copy=True)
        mb_prev = np.array(self.m[f'b{idx}'], copy=True)

        self.m[f'W{idx}'] = self.m[f'W{idx}'] * mu - self.learning_rate * grads[f'dW{idx}']
        self.m[f'b{idx}'] = self.m[f'b{idx}'] * mu - self.learning_rate * grads[f'db{idx}']
    
        w_update = -mu * mW_prev + (1 + mu) * self.m[f'W{idx}']
        b_update = -mu * mb_prev + (1 + mu) * self.m[f'b{idx}']

        layers[idx].W += w_update
        layers[idx].b += b_update

class Adagrad(Optimizer):
    '''Adagrad: https://jmself.learning_rate.org/papers/volume12/duchi11a/duchi11a.pdf'''

    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.v = dict()

    def config(self, layers):
        for i in layers.keys():
            self.v[f'W{i}'] = 0
            self.v[f'b{i}'] = 0
    
    def optimize(self, idx, layers, grads, epoch_num, steps):
        self.v[f'W{idx}'] += grads[f'dW{idx}'] **2 
        self.v[f'b{idx}'] += grads[f'db{idx}'] **2

        w_update = - self.learning_rate * grads[f'dW{idx}'] / (np.sqrt(self.v[f'W{idx}'] + self.epsilon))
        b_update = - self.learning_rate * grads[f'db{idx}'] / (np.sqrt(self.v[f'b{idx}']+ self.epsilon))

        layers[idx].W += w_update
        layers[idx].b += b_update

class RMSprop(Optimizer):
    def __init__(self, decay_rate=0.9, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = dict()

    def config(self, layers):
        for i in layers.keys():
            self.cache[f'W{i}'] = 0
            self.cache[f'b{i}'] = 0

    def optimize(self, idx, layers, grads, epoch_num, steps):
        self.cache[f'W{idx}'] = self.decay_rate * self.cache[f'W{idx}'] + (1 - self.decay_rate) * grads[f'dW{idx}'] **2 
        self.cache[f'b{idx}'] = self.decay_rate * self.cache[f'b{idx}'] + (1 - self.decay_rate) * grads[f'db{idx}'] **2
        
        w_update = - self.learning_rate * grads[f'dW{idx}'] / (np.sqrt(self.cache[f'W{idx}'] + self.epsilon))
        b_update = - self.learning_rate * grads[f'db{idx}'] / (np.sqrt(self.cache[f'b{idx}']+ self.epsilon))

        layers[idx].W += w_update
        layers[idx].b += b_update


class Adam(Optimizer):
    '''One of the most popular first-order gradient descent algorithms with momentum estimate
        terms : https://arxiv.org/pdf/1412.6980.pdf'''
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                 weight_decay=False, gamma_init=1e-5, decay_rate=0.8, demon=False, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2 
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        if self.weight_decay:
            self.gamma_init = gamma_init
            self.decay_rate = decay_rate
        self.demon = demon
        self.m = dict()  # first moment estimate 
        self.v = dict()  # second raw moment estimate 

    def config(self, layers):
        for i in layers.keys():
            self.m[f'W{i}'] = 0
            self.m[f'b{i}'] = 0
            self.v[f'W{i}'] = 0
            self.v[f'b{i}'] = 0

    def optimize(self, idx, layers, grads, epoch_num, steps): 
        dW = grads[f'dW{idx}']
        db = grads[f'db{idx}']
        if self.demon:
            p_t = 1 - epoch_num / self.epochs
            beta1 = self.beta1 * (p_t / (1 - self.beta1 + self.beta1 * p_t))
        else:
            beta1 = self.beta1

        # weights
        self.m[f'W{idx}'] = beta1 * self.m[f'W{idx}'] + (1 - beta1) * dW
        self.v[f'W{idx}'] = self.beta2 * self.v[f'W{idx}'] + (1 - self.beta2) * dW ** 2 
        
        # biases
        self.m[f'b{idx}'] = beta1 * self.m[f'b{idx}'] + (1 - beta1) * db
        self.v[f'b{idx}'] = self.beta2 * self.v[f'b{idx}'] + (1 - self.beta2) * db ** 2 

        # take timestep into account
        mt_w  = self.m[f'W{idx}'] / (1 - beta1 ** steps)
        vt_w = self.v[f'W{idx}'] / (1 - self.beta2 ** steps)

        mt_b  = self.m[f'b{idx}'] / (1 - beta1 ** steps)
        vt_b = self.v[f'b{idx}'] / (1 - self.beta2 ** steps)

        w_update = - self.learning_rate * mt_w / (np.sqrt(vt_w) + self.epsilon)
        b_update = - self.learning_rate * mt_b / (np.sqrt(vt_b) + self.epsilon)
        
        if self.weight_decay:
            gamma = self.gamma_init * self.decay_rate ** int(epoch_num / 5) 
            w_update = - self.learning_rate * mt_w / ((np.sqrt(vt_w) + self.epsilon) + gamma * layers[idx].W) 
            b_update = - self.learning_rate * mt_b / ((np.sqrt(vt_b) + self.epsilon) + gamma * layers[idx].b)

        layers[idx].W += w_update
        layers[idx].b += b_update

class DemonAdam(Adam):  
    '''Decaying Momentum in Adam: https://arxiv.org/pdf/1910.04952v3.pdf'''
    def __init__(self, learning_rate, beta1_init=0.9, **kwargs):
        super().__init__(**kwargs)
        self.beta1_init = beta1_init

    def optimize(self, idx, layers, grads, epoch_num, steps):
        p_t = 1 - epoch_num / self.epochs
        beta1 = self.beta1_init * (p_t / (1 - self.beta1_init + self.beta1_init * p_t))
        
        self.m[f'W{idx}'] = beta1 * self.m[f'W{idx}'] + (1 - beta1) * grads[f'dW{idx}']
        self.v[f'W{idx}'] = self.beta2 * self.v[f'W{idx}'] + (1 - self.beta2) * grads[f'dW{idx}'] ** 2 

        self.m[f'b{idx}'] = beta1 * self.m[f'b{idx}'] + (1 - beta1) * grads[f'db{idx}']
        self.v[f'b{idx}'] = self.beta2 * self.v[f'b{idx}'] + (1 - self.beta2) * grads[f'db{idx}'] ** 2 

        mt_w  = self.m[f'W{idx}'] / (1 - beta1 ** steps)
        vt_w = self.v[f'W{idx}'] / (1 - self.beta2 ** steps)

        mt_b  = self.m[f'b{idx}'] / (1 - beta1 ** steps)
        vt_b = self.v[f'b{idx}'] / (1 - self.beta2 ** steps)

        w_update = - self.learning_rate * mt_w / (np.sqrt(vt_w) + self.epsilon)
        b_update = - self.learning_rate * mt_b / (np.sqrt(vt_b) + self.epsilon)

        layers[idx].W += w_update
        layers[idx].b += b_update

class Nadam(Adam):
    ''' Nesterov Momentum + Adam http://cs229.stanford.edu/proj2015/054_report.pdf'''
    def __init__(self, learning_rate, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate

    def optimize(self, idx, layers, grads, epoch_num, steps): 
        dW = grads[f'dW{idx}']
        db = grads[f'db{idx}']

        if self.demon:
            p_t = 1 - epoch_num / self.epochs
            beta1 = self.beta1 * (p_t / (1 - self.beta1 + self.beta1 * p_t))
        else:
            beta1 = self.beta1

        # weights
        self.m[f'W{idx}'] = beta1 * self.m[f'W{idx}'] + (1 - beta1) * dW
        self.v[f'W{idx}'] = self.beta2 * self.v[f'W{idx}'] + (1 - self.beta2) * dW ** 2 
        
        # biases
        self.m[f'b{idx}'] = beta1 * self.m[f'b{idx}'] + (1 - beta1) * db
        self.v[f'b{idx}'] = self.beta2 * self.v[f'b{idx}'] + (1 - self.beta2) * db ** 2 

        # take timestep into account
        mt_w  = self.m[f'W{idx}'] / (1 - beta1 ** steps)
        vt_w = self.v[f'W{idx}'] / (1 - self.beta2 ** steps)

        mt_b  = self.m[f'b{idx}'] / (1 - beta1 ** steps)
        vt_b = self.v[f'b{idx}'] / (1 - self.beta2 ** steps)

        if self.weight_decay:
            gamma = self.gamma_init * self.decay_rate ** int(epoch_num / 5) 
            w_update = - self.learning_rate / (np.sqrt(vt_w) + self.epsilon + gamma * layers[idx].W) * (beta1 * mt_w + (1 - beta1) *  dW / (1 - beta1 ** steps))
            b_update = - self.learning_rate / (np.sqrt(vt_b) + self.epsilon + gamma * layers[idx].b) * (beta1 * mt_b + (1 - beta1) *  db / (1 - beta1 ** steps))
        else:
            w_update = - self.learning_rate / (np.sqrt(vt_w) + self.epsilon) * (beta1 * mt_w + (1 - beta1) *  dW / (1 - beta1 ** steps))
            b_update = - self.learning_rate / (np.sqrt(vt_b) + self.epsilon) * (beta1 * mt_b + (1 - beta1) *  db / (1 - beta1 ** steps))

        layers[idx].W += w_update
        layers[idx].b += b_update



class Adamax(Adam):
    def __init__(self, learning_rate, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate
    
    def optimize(self, idx, layers, grads, epoch_num, steps):
        if self.demon:
            p_t = 1 - epoch_num / self.epochs
            beta1 = self.beta1 * (p_t / (1 - self.beta1 + self.beta1 * p_t))
        else:
            beta1 = self.beta1

        self.m[f'W{idx}'] = beta1 * self.m[f'W{idx}'] + (1 - beta1) * grads[f'dW{idx}']                
        self.v[f'W{idx}'] = np.maximum(self.beta2 * self.v[f'W{idx}'],  abs(grads[f'dW{idx}']))
        self.m[f'b{idx}'] = beta1 * self.m[f'b{idx}'] + (1 - beta1) * grads[f'db{idx}']
        self.v[f'b{idx}'] = np.maximum(self.beta2 * self.v[f'b{idx}'],  abs(grads[f'db{idx}']))

        mt_w  = self.m[f'W{idx}'] / (1 - beta1 ** steps)
        vt_w = self.v[f'W{idx}'] / (1 - self.beta2 ** steps)

        mt_b  = self.m[f'b{idx}'] / (1 - beta1 ** steps)
        vt_b = self.v[f'b{idx}'] / (1 - self.beta2 ** steps)
        assert steps != 0  # or else it will divide by 0 

        if self.weight_decay:
            gamma = self.gamma_init * self.decay_rate ** int(epoch_num / 5) 
            w_update = - (self.learning_rate / (1 - beta1 ** steps )) * mt_w / (vt_w + self.epsilon + gamma * layers[idx].W)
            b_update = - (self.learning_rate / (1 - beta1 ** steps )) * mt_b / (vt_b + self.epsilon + gamma * layers[idx].b)
        else:
            w_update = - (self.learning_rate / (1 - beta1 ** steps )) * mt_w / (vt_w + self.epsilon)
            b_update = - (self.learning_rate / (1 - beta1 ** steps )) * mt_b / (vt_b + self.epsilon) 

        layers[idx].W += w_update
        layers[idx].b += b_update

class AdamW(Adam):  # works best (or sometimes straight up breaks otherwise) with a decaying learning rate 
    '''Adam with decoupled weight decay: https://arxiv.org/pdf/1711.05101v3.pdf'''
    def __init__(self, learning_rate, gamma_init=1e-5, decay_rate=0.8, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.gamma_init = gamma_init
        self.decay_rate = decay_rate
    
    def optimize(self, idx, layers, grads, epoch_num, steps):
        gamma = self.gamma_init * self.decay_rate ** int(epoch_num / 5) 
        dW = grads[f'dW{idx}']
        db = grads[f'db{idx}']
        self.m[f'W{idx}'] = self.beta1 * self.m[f'W{idx}'] + (1 - self.beta1) * dW
        self.v[f'W{idx}'] = self.beta2 * self.v[f'W{idx}'] + (1 - self.beta2) * dW ** 2 

        self.m[f'b{idx}'] = self.beta1 * self.m[f'b{idx}'] + (1 - self.beta1) * db
        self.v[f'b{idx}'] = self.beta2 * self.v[f'b{idx}'] + (1 - self.beta2) * db ** 2 

        mt_w  = self.m[f'W{idx}'] / (1 - self.beta1 ** steps)
        vt_w = self.v[f'W{idx}'] / (1 - self.beta2 ** steps)

        mt_b  = self.m[f'b{idx}'] / (1 - self.beta1 ** steps)
        vt_b = self.v[f'b{idx}'] / (1 - self.beta2 ** steps)

        w_update = - self.learning_rate * mt_w / ((np.sqrt(vt_w) + self.epsilon) + gamma * layers[idx].W) 
        b_update = - self.learning_rate * mt_b / ((np.sqrt(vt_b) + self.epsilon) + gamma * layers[idx].b)

        layers[idx].W += w_update
        layers[idx].b += b_update

class QHAdam(Adam):
    '''Replacing momentum estimators in Adam with quasi-hyperbolic terms:
            https://arxiv.org/pdf/1810.06801.pdf'''
    def __init__(self, v1=0.7, v2=1, **kwargs):
        super().__init__(**kwargs)
        self.v1 = v1
        self.v2 = v2

    def optimize(self, idx, layers, grads, epoch_num, steps):
        dW = grads[f'dW{idx}']
        db = grads[f'db{idx}']

        if self.demon:
            p_t = 1 - epoch_num / self.epochs
            beta1 = self.beta1 * (p_t / (1 - self.beta1 + self.beta1 * p_t))
        else:
            beta1 = self.beta1

        
        self.m[f'W{idx}'] = beta1 * self.m[f'W{idx}'] + (1 - beta1) * dW
        self.v[f'W{idx}'] = self.beta2 * self.v[f'W{idx}'] + (1 - self.beta2) * dW ** 2 

        self.m[f'b{idx}'] = beta1 * self.m[f'b{idx}'] + (1 - beta1) * db
        self.v[f'b{idx}'] = self.beta2 * self.v[f'b{idx}'] + (1 - self.beta2) * db ** 2 

        mt_w  = self.m[f'W{idx}'] / (1 - beta1 ** steps)
        vt_w = self.v[f'W{idx}'] / (1 - self.beta2 ** steps)

        mt_b  = self.m[f'b{idx}'] / (1 - beta1 ** steps)
        vt_b = self.v[f'b{idx}'] / (1 - self.beta2 ** steps)

        # Identical to Adam until here 

        if self.weight_decay:
            gamma = self.gamma_init * self.decay_rate ** int(epoch_num / 5) 
            w_update = - self.learning_rate * ((1-self.v1) * dW + self.v1 * mt_w) / (np.sqrt((1-self.v2)* dW **2 + self.v2 * vt_w) + self.epsilon + gamma * layers[idx].W)
            b_update = - self.learning_rate * ((1-self.v1) * db + self.v1 * mt_b) / (np.sqrt((1-self.v2)* db **2 + self.v2 * vt_b) + self.epsilon + gamma * layers[idx].b)
        else:
            w_update = - self.learning_rate * ((1-self.v1) * dW + self.v1 * mt_w) / (np.sqrt((1-self.v2)* dW **2 + self.v2 * vt_w) + self.epsilon)
            b_update = - self.learning_rate * ((1-self.v1) * db + self.v1 * mt_b) / (np.sqrt((1-self.v2)* db **2 + self.v2 * vt_b) + self.epsilon)

        assert w_update.shape == layers[idx].W.shape
        assert b_update.shape == layers[idx].b.shape
        layers[idx].W += w_update
        layers[idx].b += b_update

class QHM(Adam):
    '''Same paper as QHAdam https://arxiv.org/pdf/1810.06801.pdf'''
    def __init__(self, beta=0.999, v_=0.7, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.v_ = v_ 

    def optimize(self, idx, layers, grads, epoch_num, steps):

        self.v[f'W{idx}'] = self.v[f'W{idx}'] * self.beta + (1 - self.beta)  * grads[f'dW{idx}']
        self.v[f'b{idx}'] = self.v[f'b{idx}'] * self.beta + (1 - self.beta)  * grads[f'db{idx}']

        w_update = - self.learning_rate * ((1-self.v_) * grads[f'dW{idx}'] + self.v_ * self.v[f'W{idx}'])
        b_update = - self.learning_rate * ((1-self.v_) * grads[f'db{idx}'] + self.v_ * self.v[f'b{idx}'])

        layers[idx].W += w_update
        layers[idx].b += b_update

class Adadelta(Adam):
    '''Adaptive learning rate method without the need to explicitly set a learning rate : https://arxiv.org/pdf/1212.5701.pdf'''
    def __init__(self, gamma=0.9, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma 


    def optimize(self, idx, layers, grads, epoch_num, steps):

        # squared grad var
        self.v[f'W{idx}'] = self.gamma * self.v[f'W{idx}'] + (1 - self.gamma) * grads[f'dW{idx}'] ** 2
        self.v[f'b{idx}'] = self.gamma * self.v[f'b{idx}'] + (1 - self.gamma) * grads[f'db{idx}'] ** 2

        w_update = - np.sqrt(self.m[f'W{idx}'] + self.epsilon) / np.sqrt(self.v[f'W{idx}'] + self.epsilon) * grads[f'dW{idx}'] 
        b_update = - np.sqrt(self.m[f'b{idx}'] + self.epsilon) / np.sqrt(self.v[f'b{idx}'] + self.epsilon) * grads[f'db{idx}'] 

        # grad updates var 
        self.m[f'W{idx}'] = self.gamma * self.m[f'W{idx}']  + (1 - self.gamma) * w_update ** 2
        self.m[f'b{idx}'] = self.gamma * self.m[f'b{idx}']  + (1 - self.gamma) * b_update ** 2

        layers[idx].W += w_update
        layers[idx].b += b_update
    

