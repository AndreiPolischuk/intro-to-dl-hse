import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        
        N = 1 if input.ndim == 1 else input.shape[1]
        
        loss = 1 / (input.shape[0] * N) * (np.sum((input - target) ** 2))
        
        return loss
        
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return super().compute_output(input, target)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        
        N = 1 if input.ndim == 1 else input.shape[1]
        
        return 2 / (N * input.shape[0]) * (input - target)
        
        return 
        
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return super().compute_grad_input(input, target)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """

    # ---------- forward ----------
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits, shape (B, C)
        :param target: class indices (B,)  *or* one‑hot matrix (B, C)
        :return: scalar mean loss
        """
        self.log_softmax = LogSoftmax()   # reuse the module you already wrote
        self._log_probs = None            # cache needed for backward
        
        # 1) log‑softmax once, numerically stable
        self._log_probs = self.log_softmax(input)          # (B, C)

        # 2) pick correct log‑probs
        if target.ndim == 1:                               # integer labels
            loss_sample = -self._log_probs[
                np.arange(input.shape[0]), target
            ]
        else:                                              # one‑hot mask
            loss_sample = -np.sum(self._log_probs * target, axis=1)

        # 3) mean over batch
        return (np.mean(loss_sample))
        
        
        
        return super().compute_output(input, target)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        
        B, C = input.shape
        if target.ndim == 1:                       # целочисленные метки
            mask = np.zeros((B, C), dtype=input.dtype)
            mask[np.arange(B), target] = 1.0
        else:                                      # уже one‑hot
            mask = target.astype(input.dtype)

        # ----- 2. Softmax (численно устойчивый) ------------------------
        shifted = input - np.max(input, axis=1, keepdims=True)  # (B, C)
        exp_shifted = np.exp(shifted)
        softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        # ----- 3. Градиент для mean‑reduction --------------------------
        grad_input = (softmax - mask) / B          # (B, C)

        return grad_input
        
        

        
        return super().compute_grad_input(input, target)
