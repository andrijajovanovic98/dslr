# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#   logreg_train.py                                    :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2025/02/18 11:15:36 by iberegsz          #+#    #+#              #
#   Updated: 2025/04/02 11:37:56 by iberegsz         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    """
    Compute the sigmoid function.

    Args:
        z (numpy.ndarray): Input value.

    Rises:
        ValueError: If the input value is not a numpy array.
        TypeError: If the input value is not a number.

    Returns:
        numpy.ndarray: Sigmoid of the input value.
    """
    try:
        return 1 / (1 + np.exp(-z))
    except Exception as e:
        raise Exception(f"Error in sigmoid function: {e}")


def compute_loss(X, y, theta):
    """
    Compute the loss function for logistic regression.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target variable.
        theta (numpy.ndarray): Model weights.

    Raises:
        ValueError: If the input matrices have incompatible shapes.
        TypeError: If the input matrices are not numpy arrays.
        Exception: If any other unexpected error occurs.

    Returns:
        float: Computed loss.
    """
    try:
        m = len(y)
        h = sigmoid(X.dot(theta))
        return (-1 / m) * np.sum(y * np.log(h + 1e-8) +
                                 (1 - y) * np.log(1 - h + 1e-8))
    except Exception as e:
        raise Exception(f"Error in compute_loss function: {e}")


def gradient_descent(X, y, theta, learning_rate, num_iterations):
    """
    Perform gradient descent to learn theta.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target variable.
        theta (numpy.ndarray): Initial model weights.
        learning_rate (float): Learning rate.
        num_iterations (int): Number of iterations.

    Raises:
        ValueError: If the input matrices have incompatible shapes.
        TypeError: If the input matrices are not numpy arrays.
        Exception: If any other unexpected error occurs.

    Returns:
        tuple: Updated model weights and loss history.
    """
    try:
        m = len(y)
        losses = []
        for i in range(num_iterations):
            h = sigmoid(X.dot(theta))
            gradient = (1 / m) * X.T.dot(h - y)
            theta -= learning_rate * gradient

            loss = compute_loss(X, y, theta)
            if np.isnan(loss):
                print(f"Loss became NaN at iteration {i}. ", end="")
                print("Check data/learning rate.")
                break
            losses.append(loss)

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

        return theta, losses
    except Exception as e:
        raise Exception(f"Error in gradient_descent function: {e}")


def load_data(filepath):
    """
    Load and preprocess the dataset.

    Args:
        filepath (str): Path to the dataset file.

    Raises:
        FileNotFoundError: If the dataset file is not found.
        ValueError: If NaN values are detected after preprocessing.
        Exception: If any other unexpected error occurs during loading.

    Returns:
        tuple: Feature matrix and target variable.
    """
    try:
        data = pd.read_csv(filepath)

        X = data.drop(['Hogwarts House', 'First Name',
                      'Last Name', 'Birthday'], axis=1)

        X = pd.get_dummies(X, columns=['Best Hand'], drop_first=False)

        X = X.apply(pd.to_numeric, errors='coerce')
        X.fillna(X.mean(), inplace=True)

        if X.isnull().any().any():
            raise ValueError("NaN values detected after preprocessing!")

        y = pd.get_dummies(data['Hogwarts House'])

        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        joblib.dump(scaler, "models/scaler.pkl")

        X = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))

        return X, y.values
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error in load_data function: {e}")


def train_logistic_regression(X, y):
    """
    Train a logistic regression model using gradient descent.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target variable.

    Raises:
        ValueError: If NaN values are detected in the dataset after
                    preprocessing.
        Exception: If any other unexpected error occurs during training.
        FileNotFoundError: If the dataset file is not found.

    Returns:
        numpy.ndarray: Trained model weights.
    """
    try:
        theta = np.zeros((X.shape[1], y.shape[1]))
        print(f"theta shape: {theta.shape}, theta type: {type(theta)}")

        learning_rate = 0.01
        num_iterations = 1000

        theta, losses = gradient_descent(
            X, y, theta, learning_rate, num_iterations)
        np.savetxt("models/weights.txt", theta)
        return theta
    except Exception as e:
        raise Exception(f"Error in train_logistic_regression function: {e}")


def main():
    """
    Main function to train the logistic regression model.

    Args:
        None

    Raises:
        ValueError: If NaN values are detected in the dataset after
                    preprocessing.
        FileNotFoundError: If the dataset file is not found.
        Exception: If any other unexpected error occurs during training.

    Returns:
        None
    """
    try:
        if len(sys.argv) > 1:
            data_path = sys.argv[1]
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, "../data/dataset_train.csv")

        X, y = load_data(data_path)
        theta = train_logistic_regression(X, y)
        print(f"Trained model weights: {theta}")
    except Exception as e:
        print(f"Error in main function: {e}")


if __name__ == "__main__":
    main()
