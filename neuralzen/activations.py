#activations.py
import math

class Activations:
    @staticmethod
    def ReLu(x: float) -> float:
        """ReLU Activation: max(0, x)"""
        return max(0, x)

    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid Activation: 1 / (1 + e^(-x))"""
        return 1.0 / (1 + math.exp(-x))
