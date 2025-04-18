# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#   logreg_train_bonus.py                              :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2025/02/19 22:49:58 by iberegsz          #+#    #+#              #
#   Updated: 2025/04/02 11:35:50 by iberegsz         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

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
    except ValueError as ve:
        raise ValueError(f"ValueError in compute_loss: {ve}")
    except TypeError as te:
        raise TypeError(f"TypeError in compute_loss: {te}")
    except Exception as e:
        raise Exception(f"Unexpected error in compute_loss: {e}")


def batch_gradient_descent(X, y, theta, learning_rate, num_iterations):
    """
    Perform batch gradient descent for logistic regression.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target variable.
        theta (numpy.ndarray): Model weights.
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
        for _ in range(num_iterations):
            h = sigmoid(X.dot(theta))
            gradient = (1 / m) * X.T.dot(h - y)
            theta -= learning_rate * gradient
            loss = compute_loss(X, y, theta)
            losses.append(loss)
        return theta, losses
    except ValueError as ve:
        raise ValueError(f"ValueError in batch_gradient_descent: {ve}")
    except TypeError as te:
        raise TypeError(f"TypeError in batch_gradient_descent: {te}")
    except Exception as e:
        raise Exception(f"Unexpected error in batch_gradient_descent: {e}")


def mini_batch_gradient_descent(
        X,
        y,
        theta,
        learning_rate,
        num_epochs,
        batch_size=32):
    """
    Perform mini-batch gradient descent for logistic regression.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target variable.
        theta (numpy.ndarray): Model weights.
        learning_rate (float): Learning rate.
        num_epochs (int): Number of epochs.
        batch_size (int): Size of each mini-batch."

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
        for epoch in range(num_epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                h = sigmoid(X_batch.dot(theta))
                gradient = (1 / batch_size) * X_batch.T.dot(h - y_batch)
                theta -= learning_rate * gradient
            loss = compute_loss(X, y, theta)
            losses.append(loss)
        return theta, losses
    except ValueError as ve:
        raise ValueError(f"ValueError in mini_batch_gradient_descent: {ve}")
    except TypeError as te:
        raise TypeError(f"TypeError in mini_batch_gradient_descent: {te}")
    except Exception as e:
        raise Exception(
            f"Unexpected error in mini_batch_gradient_descent: {e}")


def stochastic_gradient_descent(X, y, theta, learning_rate, num_epochs):
    """
    Perform stochastic gradient descent for logistic regression.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target variable.
        theta (numpy.ndarray): Model weights.
        learning_rate (float): Learning rate.
        num_epochs (int): Number of epochs."

    Raises:
        ValueError: If the input matrices have incompatible shapes.
        TypeError: If the input matrices are not numpy arrays.
        Exception: If any other unexpected error occurs.

    Returns:
        tuple: Updated model weights and loss history.
    """
    try:
        return mini_batch_gradient_descent(
            X, y, theta, learning_rate, num_epochs, batch_size=1)
    except ValueError as ve:
        raise ValueError(f"ValueError in stochastic_gradient_descent: {ve}")
    except TypeError as te:
        raise TypeError(f"TypeError in stochastic_gradient_descent: {te}")
    except Exception as e:
        raise Exception(
            f"Unexpected error in stochastic_gradient_descent: {e}")


def load_data(filepath):
    """
    Load and preprocess the dataset.

    Args:
        filepath (str): Path to the dataset CSV file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: If any other unexpected error occurs.

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
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        joblib.dump(scaler, "models/scaler.pkl")
        X = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))
        y = pd.get_dummies(data['Hogwarts House']).values
        return X, y
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(f"FileNotFoundError in load_data: {fnfe}")
    except ValueError as ve:
        raise ValueError(f"ValueError in load_data: {ve}")
    except TypeError as te:
        raise TypeError(f"TypeError in load_data: {te}")
    except Exception as e:
        raise Exception(f"Unexpected error in load_data: {e}")


def train_model(
        optimizer='batch',
        learning_rate=0.01,
        num_epochs=1000,
        batch_size=32):
    """
    Train a logistic regression model using the specified optimizer.

    Args:
        optimizer (str): Optimizer to use ('batch', 'mini-batch', 'sgd').
        learning_rate (float): Learning rate.
        num_epochs (int): Number of epochs.
        batch_size (int): Size of each mini-batch."

    Raises:
        ValueError: If an unsupported optimizer is specified.
        Exception: If any other unexpected error occurs.

    Returns:
        None
    """
    try:
        X, y = load_data("data/dataset_train.csv")
        theta = np.zeros((X.shape[1], y.shape[1]))
        if optimizer == 'batch':
            theta, _ = batch_gradient_descent(
                X, y, theta, learning_rate, num_epochs)
        elif optimizer == 'mini-batch':
            theta, _ = mini_batch_gradient_descent(
                X, y, theta, learning_rate, num_epochs, batch_size)
        elif optimizer == 'sgd':
            theta, _ = stochastic_gradient_descent(
                X, y, theta, learning_rate, num_epochs)
        else:
            raise ValueError("Unsupported optimizer.")
        np.savetxt("models/weights.txt", theta)
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(f"FileNotFoundError in train_model: {fnfe}")
    except ValueError as ve:
        raise ValueError(f"ValueError in train_model: {ve}")
    except TypeError as te:
        raise TypeError(f"TypeError in train_model: {te}")
    except Exception as e:
        raise Exception(f"Unexpected error in train_model: {e}")


def main():
    """
    Main function to execute the training process.
    It loads the dataset, trains the logistic regression model using
    different optimizers, and saves the model weights.

    Args:
        None

    Raises:
        ValueError: If an unsupported optimizer is specified.
        FileNotFoundError: If the dataset file does not exist.
        Exception: If any other unexpected error occurs during training.

    Returns:
        None
    """
    try:
        print("Training with batch gradient descent...")
        train_model(
            optimizer='batch',
            learning_rate=0.01,
            num_epochs=1000,
            batch_size=32)

        # print("Training with mini-batch gradient descent...")
        # train_model(
        #     optimizer='mini-batch',
        #     learning_rate=0.01,
        #     num_epochs=1000,
        #     batch_size=32)

        # print("Training with stochastic gradient descent...")
        # train_model(
        #     optimizer='sgd',
        #     learning_rate=0.01,
        #     num_epochs=1000,
        #     batch_size=32)
    except ValueError as ve:
        raise ValueError(f"ValueError in main: {ve}")
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(f"FileNotFoundError in main: {fnfe}")
    except Exception as e:
        raise Exception(f"Unexpected error in main: {e}")


if __name__ == "__main__":
    main()
