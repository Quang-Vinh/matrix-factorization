import math
import numba as nb
import numpy as np
import pandas as pd

from .kernels import (
    kernel_linear,
    kernel_sigmoid,
    kernel_rbf,
    kernel_linear_sgd_update,
    kernel_sigmoid_sgd_update,
    kernel_rbf_sgd_update,
)
from .recommender_base import RecommenderBase

from typing import Tuple, Union


class KernelMF(RecommenderBase):
    """ 
    Kernel Matrix Factorization. Finds the thin matrices P and Q such that P * Q^T give a good low rank approximation to the user-item 
    ratings matrix A based on RMSE. This is different from SVD despite the name as unlike SVD there is no constraint for matrices P and Q to have mutually
    orthogonal columns. 

    Arguments:
        n_factors {int} -- The number of latent factors in matrices P and Q (default: {100})
        n_epochs {int} -- Number of epochs to train for (default: {100})
        kernel {str} -- Kernel function to use between user and item features. Options are 'linear', 'logistic' or 'rbf'. (default: {'linear'})
        gamma {str or float} -- Kernel coefficient for 'rbf'. Ignored by other kernels. If 'auto' is used then will be set to 1/n_factors. (default: 'auto')
        reg {float} -- Regularization parameter lambda for Tikhonov regularization (default: {0.01})
        lr {float} -- Learning rate alpha for gradient optimization step (default: {0.01})
        init_mean {float} -- Mean of normal distribution to use for initializing parameters (default: {0})
        init_sd {float} -- Standard deviation of normal distribution to use for initializing parameters (default: {0.1})
        min_rating {int} -- Smallest rating possible (default: {0})
        max_rating {int} -- Largest rating possible (default: {5})
        verbose {str} -- Verbosity when fitting. Values possible are 0 to not print anything, 1 to print fitting model (default: {1})

    Attributes:
        n_users {int} -- Number of users
        n_items {int} -- Number of items
        global_mean {float} -- Global mean of all ratings
        user_biases {numpy array} -- User bias vector of shape (n_users, 1)
        item_biases {numpy array} -- Item bias vector of shape (n_items, i)
        user_features {numpy array} -- Decomposed P matrix of user features of shape (n_users, n_factors)
        item_features {numpy array} -- Decomposed Q matrix of item features of shape (n_items, n_factors)
        user_id_map {dict} -- Mapping of user ids to assigned integer ids
        item_id_map {dict} -- Mapping of item ids to assigned integer ids
        train_rmse -- Training rmse values
        predictions_possible {list} -- Boolean vector of whether both user and item were known for prediction. Only available after calling predict
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 100,
        kernel: str = "linear",
        gamma: Union[str, float] = "auto",
        reg: float = 1,
        lr: float = 0.01,
        init_mean: float = 0,
        init_sd: float = 0.1,
        min_rating: int = 0,
        max_rating: int = 5,
        verbose: int = 1,
    ):
        if kernel not in ("linear", "sigmoid", "rbf"):
            raise ValueError("Kernel must be one of linear, sigmoid, or rbf")

        super().__init__(min_rating=min_rating, max_rating=max_rating, verbose=verbose)

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.kernel = kernel
        self.gamma = 1 / n_factors if gamma == "auto" else gamma
        self.reg = reg
        self.lr = lr
        self.init_mean = init_mean
        self.init_sd = init_sd
        return

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """ 
        Decompose user-item rating matrix into thin matrices P and Q along with user and item bias vectors

        Arguments:
            X {pandas DataFrame} -- Dataframe containing columns user_id, item_id 
            y {pandas Series} -- Series containing ratings
        """
        X = self._preprocess_data(X=X, y=y, type="fit")
        self.global_mean = X["rating"].mean()

        # Initialize vector bias parameters
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)

        # Initialize latent factor parameters of matrices P and Q
        self.user_features = np.random.normal(
            self.init_mean, self.init_sd, (self.n_users, self.n_factors)
        )
        self.item_features = np.random.normal(
            self.init_mean, self.init_sd, (self.n_items, self.n_factors)
        )

        # Perform stochastic gradient descent
        (
            self.user_features,
            self.item_features,
            self.user_biases,
            self.item_biases,
            self.train_rmse,
        ) = _sgd(
            X=X.to_numpy(),
            global_mean=self.global_mean,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            user_features=self.user_features,
            item_features=self.item_features,
            n_epochs=self.n_epochs,
            kernel=self.kernel,
            gamma=self.gamma,
            lr=self.lr,
            reg=self.reg,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
            verbose=self.verbose,
        )

        return self

    def predict(self, X: pd.DataFrame, bound_ratings: bool = True) -> list:
        """
        Predict ratings for given users and items

        Arguments:
            X {pd.DataFrame} -- Dataframe containing columns user_id and item_id
            bound_ratings (bool): Whether to bound ratings in range [min_rating, max_rating] (default: True)

        Returns:
            predictions [list] -- List containing rating predictions of all user, items in same order as input X
        """
        # If empty return empty list
        if X.shape[0] == 0:
            return []

        X = self._preprocess_data(X=X, type="predict")

        # Get predictions
        predictions, predictions_possible = _predict(
            X=X.to_numpy(),
            global_mean=self.global_mean,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            user_features=self.user_features,
            item_features=self.item_features,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
            kernel=self.kernel,
            gamma=self.gamma,
            bound_ratings=bound_ratings,
        )

        self.predictions_possible = predictions_possible
        return predictions

    def update_users(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        lr: float = 0.01,
        n_epochs: int = 20,
        verbose: int = 0,
    ):
        """
        Update P user features matrix with new/updated user-item ratings information using SGD. Only the user parameters corresponding for the
        new/updated users will be updated and item parameters will be left alone.

        Note: If updating old users then pass all user-item ratings for old users and not just modified ratings

        Args:
            X (pd.DataFrame): Dataframe containing columns user_id, item_id 
            y (pd.DataFrame): Series containing ratings
            lr (float, optional): Learning rate alpha for gradient optimization step
            n_epochs (int, optional): Number of epochs to run SGD. Defaults to 20.
            verbose (int, optional): Verbosity when updating, 0 for nothing and 1 for training messages. Defaults to 0.
        """
        X, known_users, new_users = self._preprocess_data(X=X, y=y, type="update")
        n_new_users = len(new_users)

        # Re-initialize params for old users
        for user in known_users:
            user_index = self.user_id_map[user]

            # Initialize bias
            self.user_biases[user_index] = 0

            # Initialize latent factors vector
            self.user_features[user_index, :] = np.random.normal(
                self.init_mean, self.init_sd, (1, self.n_factors)
            )

        # Add bias parameters for new users
        self.user_biases = np.append(self.user_biases, np.zeros(n_new_users))

        # Add latent factor parameters for new users by adding rows to P matrix
        new_user_features = np.random.normal(
            self.init_mean, self.init_sd, (n_new_users, self.n_factors)
        )
        self.user_features = np.concatenate(
            (self.user_features, new_user_features), axis=0
        )

        # Estimate new parameters
        (
            self.user_features,
            self.item_features,
            self.user_biases,
            self.item_biases,
            self.train_rmse,
        ) = _sgd(
            X=X.to_numpy(),
            global_mean=self.global_mean,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            user_features=self.user_features,
            item_features=self.item_features,
            n_epochs=n_epochs,
            kernel=self.kernel,
            gamma=self.gamma,
            lr=lr,
            reg=self.reg,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
            verbose=verbose,
            update_item_params=False,
        )

        return


@nb.njit()
def _calculate_rmse(
    X: np.ndarray,
    global_mean: float,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_features: np.ndarray,
    item_features: np.ndarray,
    min_rating: float,
    max_rating: float,
    kernel: str,
    gamma: float,
):
    """
    Calculates root mean squared error for given data and model parameters

    Args:
        X (np.ndarray): Matrix with columns user, item and rating
        global_mean (float): Global mean rating
        user_biases (np.ndarray): User biases vector of shape (n_users, 1)
        item_biases (np.ndarray): Item biases vector of shape (n_items, 1)
        user_features (np.ndarray): User features matrix P of size (n_users, n_factors)
        item_features (np.ndarray): Item features matrix Q of size (n_items, n_factors)
        min_rating (float): Minimum possible rating
        max_rating (float): Maximum possible rating
        kernel (str): Kernel type. Possible options are "linear", "sigmoid" or "rbf" kernel
        gamma (float): Kernel coefficient only for "rbf" kernel

    Returns:
        rmse [float]: Root mean squared error
    """
    n_ratings = X.shape[0]
    errors = np.zeros(n_ratings)

    # Iterate through all user-item ratings and calculate error
    for i in range(n_ratings):
        user_id, item_id, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        user_bias = user_biases[user_id]
        item_bias = item_biases[item_id]
        user_feature_vec = user_features[user_id, :]
        item_feature_vec = item_features[item_id, :]

        # Calculate predicted rating for given kernel
        if kernel == "linear":
            rating_pred = kernel_linear(
                global_mean=global_mean,
                user_bias=user_bias,
                item_bias=item_bias,
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
            )

        elif kernel == "sigmoid":
            rating_pred = kernel_sigmoid(
                global_mean=global_mean,
                user_bias=user_bias,
                item_bias=item_bias,
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
                a=min_rating,
                c=max_rating - min_rating,
            )

        elif kernel == "rbf":
            rating_pred = kernel_rbf(
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
                gamma=gamma,
                a=min_rating,
                c=max_rating - min_rating,
            )

        # Calculate error
        errors[i] = rating - rating_pred

    rmse = np.sqrt(np.square(errors).mean())

    return rmse


@nb.njit()
def _sgd(
    X: np.ndarray,
    global_mean: float,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_features: np.ndarray,
    item_features: np.ndarray,
    n_epochs: int,
    kernel: str,
    gamma: float,
    lr: float,
    reg: float,
    min_rating: float,
    max_rating: float,
    verbose: int,
    update_user_params: bool = True,
    update_item_params: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Performs stochastic gradient descent to estimate parameters.

    Arguments:
        X {numpy array} -- User-item ranking matrix
        global_mean {float} -- Global mean of all ratings
        user_biases {numpy array} -- User biases vector of shape (n_users, 1)
        item_biases {numpy array} -- Item biases vector of shape (n_items, 1)
        user_features {numpy array} -- Start matrix P of user features of shape (n_users, n_factors)
        item_features {numpy array} -- Start matrix Q of item features of shape (n_items, n_factors)
        n_epochs {int} -- Number of epochs to run
        kernel {str} -- Kernel function to use between user and item features. Options are 'linear', 'logistic', and 'rbf'. 
        gamma {float} -- Kernel coefficient for 'rbf'. Ignored by other kernels. 
        lr {float} -- Learning rate alpha
        reg {float} -- Regularization parameter lambda for Frobenius norm
        min_rating {float} -- Minimum possible rating
        max_fating {float} -- Maximum possible rating
        verbose {int} -- Verbosity when fitting. 0 for nothing and 1 for printing epochs
        update_user_params {bool} -- Whether to update user parameters or not. Default is True.
        update_item_params {bool} -- Whether to update item  parameters or not. Default is True.

    Returns:
        user_features [np.ndarray] -- Updated user_features matrix P
        item_features [np.ndarray] -- Updated item_features matrix Q
        user_biases [np.ndarray] -- Updated user_biases vector
        item_biases [np.ndarray] -- Updated item_bases vector
        train_rmse [list] -- Training rmse values
    """
    train_rmse = []

    for epoch in range(n_epochs):
        # Shuffle dataset before each epoch
        np.random.shuffle(X)

        # Iterate through all user-item ratings
        for i in range(X.shape[0]):
            user_id, item_id, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

            if kernel == "linear":
                kernel_linear_sgd_update(
                    user_id=user_id,
                    item_id=item_id,
                    rating=rating,
                    global_mean=global_mean,
                    user_biases=user_biases,
                    item_biases=item_biases,
                    user_features=user_features,
                    item_features=item_features,
                    lr=lr,
                    reg=reg,
                    update_user_params=update_user_params,
                    update_item_params=update_item_params,
                )

            elif kernel == "sigmoid":
                kernel_sigmoid_sgd_update(
                    user_id=user_id,
                    item_id=item_id,
                    rating=rating,
                    global_mean=global_mean,
                    user_biases=user_biases,
                    item_biases=item_biases,
                    user_features=user_features,
                    item_features=item_features,
                    lr=lr,
                    reg=reg,
                    a=min_rating,
                    c=max_rating - min_rating,
                    update_user_params=update_user_params,
                    update_item_params=update_item_params,
                )

            elif kernel == "rbf":
                kernel_rbf_sgd_update(
                    user_id=user_id,
                    item_id=item_id,
                    rating=rating,
                    user_features=user_features,
                    item_features=item_features,
                    lr=lr,
                    reg=reg,
                    gamma=gamma,
                    a=min_rating,
                    c=max_rating - min_rating,
                    update_user_params=update_user_params,
                    update_item_params=update_item_params,
                )

        # Calculate error and print
        rmse = _calculate_rmse(
            X=X,
            global_mean=global_mean,
            user_biases=user_biases,
            item_biases=item_biases,
            user_features=user_features,
            item_features=item_features,
            min_rating=min_rating,
            max_rating=max_rating,
            kernel=kernel,
            gamma=gamma,
        )
        train_rmse.append(rmse)

        if verbose == 1:
            print("Epoch ", epoch + 1, "/", n_epochs, " -  train_rmse:", rmse)

    return user_features, item_features, user_biases, item_biases, train_rmse


@nb.njit()
def _predict(
    X: np.ndarray,
    global_mean: float,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_features: np.ndarray,
    item_features: np.ndarray,
    min_rating: int,
    max_rating: int,
    kernel: str,
    gamma: float,
    bound_ratings: bool,
) -> Tuple[list, list]:
    """ 
    Calculate predicted ratings for each user-item pair.

    Arguments:
        X {np.ndarray} -- Matrix with columns representing (user_id, item_id)
        global_mean {float} -- Global mean of all ratings
        user_biases {np.ndarray} -- User biases vector of length n_users
        item_biases {np.ndarray} -- Item biases vector of length n_items
        user_features {np.ndarray} -- User features matrix P of shape (n_users, n_factors)
        item_features {np.ndarray} -- Item features matrix Q of shape (n_items, n_factors)
        min_rating {int} -- Lowest rating possible
        max_rating {int} -- Highest rating possible
        kernel {str} -- Kernel function. Options are 'linear', 'sigmoid', and 'rbf'
        gamma {float} -- Kernel coefficient for 'rbf' only
        bound_ratings (bool): Whether to bound ratings in range [min_rating, max_rating] (default: True)

    Returns:
        predictions [np.ndarray] -- Vector containing rating predictions of all user, items in same order as input X
        predictions_possible [np.ndarray] -- Vector of whether both given user and item were contained in the data that the model was fitted on
    """
    n_factors = user_features.shape[1]
    predictions = []
    predictions_possible = []

    for i in range(X.shape[0]):
        user_id, item_id = int(X[i, 0]), int(X[i, 1])
        user_known = user_id != -1
        item_known = item_id != -1

        # Default values if user or item are not known
        user_bias = user_biases[user_id] if user_known else 0
        item_bias = item_biases[item_id] if item_known else 0
        user_feature_vec = (
            user_features[user_id, :] if user_known else np.zeros(n_factors)
        )
        item_feature_vec = (
            item_features[item_id, :] if item_known else np.zeros(n_factors)
        )

        # Calculate predicted rating given kernel
        if kernel == "linear":
            rating_pred = kernel_linear(
                global_mean=global_mean,
                user_bias=user_bias,
                item_bias=item_bias,
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
            )

        elif kernel == "sigmoid":
            rating_pred = kernel_sigmoid(
                global_mean=global_mean,
                user_bias=user_bias,
                item_bias=item_bias,
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
                a=min_rating,
                c=max_rating - min_rating,
            )

        elif kernel == "rbf":
            rating_pred = kernel_rbf(
                user_feature_vec=user_feature_vec,
                item_feature_vec=item_feature_vec,
                gamma=gamma,
                a=min_rating,
                c=max_rating - min_rating,
            )

        # Bound ratings to min and max rating range
        if bound_ratings:
            if rating_pred > max_rating:
                rating_pred = max_rating
            elif rating_pred < min_rating:
                rating_pred = min_rating

        predictions.append(rating_pred)
        predictions_possible.append(user_known and item_known)

    return predictions, predictions_possible
