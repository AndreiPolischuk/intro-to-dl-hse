import numpy as np
from typing import Tuple
from .base import Module, Optimizer


class SGD(Optimizer):
    """
    Optimizer implementing stochastic gradient descent with momentum
    """
    def __init__(self, module: Module, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param momentum: momentum coefficient (alpha)
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]

        for param, grad, m in zip(parameters, gradients, self.state['m']):
            """
            your code here ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
              - update momentum variable (m)
              - update parameter variable (param)
            hint: consider using np.add(..., out=m) for in place addition,
              i.e. we need to change original array, not its copy
            """
            
            gr = grad if self.weight_decay == 0 else grad + self.weight_decay * (param ** 2)
            
            if self.momentum != 0:
              if np.all(m == 0):
                np.add(m, gr, out=m)

              else:
                np.add(m * self.momentum,  gr, out=m) 
              
              np.add(param, - self.lr * (m), out=param)
            else:
              np.add(param, - self.lr * gr, out=param)
            
            

            #np.add(self.momentum * m, (1 - self.momentum) * grad, out = m)
            #np.add(param, - self.lr * (m) - self.weight_decay * (param ** 2), out = param)



class Adam(Optimizer):
    """
    Optimizer implementing Adam
    """
    def __init__(self, module: Module, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param betas: Adam beta1 and beta2
        :param eps: Adam eps
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]
            self.state['v'] = [np.zeros_like(param) for param in parameters]
            self.state['t'] = 0

        self.state['t'] += 1
        t = self.state['t']
        for param, grad, m, v in zip(parameters, gradients, self.state['m'], self.state['v']):
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param
            
            # Update first moment (m)
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            np.multiply(m, self.beta1, out=m)
            np.add(m, (1 - self.beta1) * grad, out=m)

            # Update second moment (v)
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            np.multiply(v, self.beta2, out=v)
            np.add(v, (1 - self.beta2) * (grad ** 2), out=v)

            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # Update parameter
            # param_t = param_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            np.subtract(param, update, out=param)
