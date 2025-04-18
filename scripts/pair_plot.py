# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#   pair_plot.py                                       :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2025/02/19 10:14:38 by iberegsz          #+#    #+#              #
#   Updated: 2025/04/02 11:36:40 by iberegsz         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_pairplot(filepath):
    """
    Generate a pair plot for the dataset.

    Args:
        filepath (str): Path to the dataset CSV file.

    Raises:
        FileNotFoundError: If the dataset file is not found.
        Exception: If any other unexpected error occurs.

    Returns:
        None
    """
    try:
        data = pd.read_csv(filepath)

        numerical_data = data.select_dtypes(include=['float64', 'int64'])

        full_data = data[['Hogwarts House'] + list(numerical_data.columns)]

        sns.pairplot(full_data, hue='Hogwarts House', palette='viridis')

        fig1 = plt.gcf()

        plt.show()

        fig1.savefig("results/pair_plot.png")
        plt.close()
        print("Pair plot saved to results/")
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """
    Main function to execute the pair plot generation.

    Args:
        None

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        Exception: If any other unexpected error occurs.

    Returns:
        None
    """
    try:
        dataset_path = sys.argv[1] if len(
            sys.argv) > 1 else "data/dataset_train.csv"
        plot_pairplot(dataset_path)
    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except Exception as e:
        print(f"An error occurred in main(): {e}")


if __name__ == "__main__":
    main()
