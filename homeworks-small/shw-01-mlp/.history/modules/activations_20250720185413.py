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
        
        return sigma * (1 - sigma)
        
        return 
        
        
        return super().compute_grad_input(input, grad_output)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
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
