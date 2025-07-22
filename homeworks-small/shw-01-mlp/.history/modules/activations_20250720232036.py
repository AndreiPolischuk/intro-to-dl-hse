import numpy as np
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    
    
    
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        
        mask = input > 0.0
        mask = mask.astype(input.dtype) 
        return input * mask
        
        return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        
        mask = input > 0.0
        mask = mask.astype(input.dtype) 
        
        return grad_output * mask
        
        return super().compute_grad_input(input, grad_output)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        
        return 1 / (1 + np.exp(-input))
        
        return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        
        sigma = 1 / (1 + np.exp(-input))
        
        return sigma * (1 - sigma) * grad_output
        
        return 
        
        
        return super().compute_grad_input(input, grad_output)


class Softmax(Module):
    """
    Applies softmax over the last dimension.
    Works for any shape; last axis is treated as 'classes'.
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        input : (batch_size, num_classes)
        return: same shape, softmax along last axis
        """
        # численно устойчивый сдвиг
        x_shift = input - input.max(axis=-1, keepdims=True)
        exps    = np.exp(x_shift)
        denom   = np.sum(exps, axis=-1, keepdims=True)   # <- axis=-1!
        self.output = exps / denom                       # сохраняем для backward
        return self.output

    def compute_grad_input(self,
                           input: np.ndarray,
                           grad_output: np.ndarray) -> np.ndarray:
        """
        grad_output has the same shape as input/output.
        Uses vectorized Jacobian‑product:  y ⊙ (g − (y·g))
        """
        # self.output уже посчитан на forward
        y = self.output
        # скалярное произведение (y · g) для каждой строки, keepdims=true
        dot = np.sum(grad_output * y, axis=-1, keepdims=True)
        grad_input = y * (grad_output - dot)
        return grad_input


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return super().compute_output(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return super().compute_grad_input(input, grad_output)
