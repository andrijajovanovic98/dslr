# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#   logreg_predict.py                                  :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2025/02/18 10:16:34 by iberegsz          #+#    #+#              #
#   Updated: 2025/04/02 11:35:36 by iberegsz         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import joblib
import numpy as np
import pandas as pd


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


def predict(filepath, weights_path):
    """
    Predict the Hogwarts House for each student in the dataset.

    Args:
        filepath (str): Path to the dataset CSV file.
        weights_path (str): Path to the weights file.

    Raises:
        FileNotFoundError: If the dataset or weights file is not found.
        Exception: If any other unexpected error occurs.

    Returns:
        None
    """
    try:
        data = pd.read_csv(filepath)

        X = data.drop(['First Name', 'Last Name', 'Birthday'], axis=1)
        X = pd.get_dummies(X, columns=['Best Hand'], drop_first=False)
        X = X.apply(pd.to_numeric, errors='coerce')
        X.fillna(X.mean(), inplace=True)

        expected_columns = ['Best Hand_Left', 'Best Hand_Right']
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0

        scaler = joblib.load("models/scaler.pkl")
        X_normalized = scaler.transform(
            X.drop(
                'Hogwarts House',
                axis=1,
                errors='ignore'))

        X = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))

        weights = np.loadtxt(weights_path)
        probabilities = sigmoid(X.dot(weights))
        houses = np.argmax(probabilities, axis=1)
        house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

        pd.DataFrame(
            {'Hogwarts House': [house_names[h] for h in houses]}
        ).to_csv(
            "results/houses.csv",
            index_label='Index'
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    except Exception as e:
        raise Exception(f"Error in predict function: {e}")


def main():
    """
    Main function to execute the prediction process.

    Args:
        None

    Raises:
        FileNotFoundError: If the dataset or weights file does not exist.
        Exception: If an unexpected error occurs during processing.

    Returns:
        None
    """
    try:
        predict("data/dataset_test.csv", "models/weights.txt")
    except FileNotFoundError as e:
        print(f"File not found error in main: {e}")
    except ValueError as e:
        print(f"Value error in main: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")


if __name__ == "__main__":
    main()
