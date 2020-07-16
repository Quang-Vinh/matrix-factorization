import math
import numba as nb
import numpy as np


@nb.njit()
def sigmoid(x: float) -> float:
    """
    Calculates sigmoid function at x

    Args:
        x (float): Input x

    Returns:
        [float]: Sigmoid at x
    """
    result = 1 / (1 + math.exp(-x))
    return result


@nb.njit()
def kernel_linear(
    global_mean: float,
    user_bias: float,
    item_bias: float,
    user_features: np.ndarray,
    item_features: np.ndarray,
) -> float:
    """
    Calculates result with a linear kernel which is essentially just the dot product

    Args:
        global_mean (float): Global mean
        user_bias (float): User bias
        item_bias (float): Item bias
        user_features (np.ndarray): Vector of user latent features 
        item_features (np.ndarray): Vector of item latent features

    Returns:
        [float]: Linear kernel result
    """
    result = global_mean + item_bias + user_bias + np.dot(user_features, item_features)
    return result


@nb.njit()
def kernel_sigmoid(
    global_mean: float,
    user_bias: float,
    item_bias: float,
    user_features: np.ndarray,
    item_features: np.ndarray,
    a: float,
    c: float,
):
    """
    Calculates result with sigmoid kernel

    Args:
        global_mean (float): Global mean
        user_bias (float): User bias
        item_bias (float): Item bias
        user_features (np.ndarray): Vector of user latent features
        item_features (np.ndarray): Vector of item latent features
        a (float): Rescaling parameter for a + c * K(u, i)
        c (float): Rescaling parameter for a + c * K(u, i)

    Returns:
        [float]: Sigmoid kernel result
    """
    linear_sum = (
        global_mean + user_bias + item_bias + np.dot(user_features, item_features)
    )
    sigmoid_result = sigmoid(linear_sum)
    result = a + c * sigmoid_result
    return result


@nb.njit()
def kernel_rbf(
    user_features: np.ndarray,
    item_features: np.ndarray,
    gamma: float,
    a: float,
    c: float,
):
    """
    Calculates result with Radial basis function kernel

    Args:
        user_features (np.ndarray): Vector of user latent features
        item_features (np.ndarray): Vector of item latent features
        gamma (float): Kernel coefficient
        a (float): Rescaling parameter for a + c * K(u, i)
        c (float): Rescaling parameter for a + c * K(u, i)

    Returns:
        [float]: RBF kernel result 
    """
    power = -gamma * np.sum(np.square(user_features - item_features))
    exp_result = math.exp(power)
    result = a + c * exp_result
    return result
