# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#   scatter_plot.py                                    :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2025/02/18 10:10:41 by iberegsz          #+#    #+#              #
#   Updated: 2025/04/02 11:36:25 by iberegsz         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_scatter(filepath):
    """
    Generate a scatter plot comparing Astronomy and Defense Against
    the Dark Arts scores.

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

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=data,
            x='Astronomy',
            y='Defense Against the Dark Arts',
            hue='Hogwarts House',
            palette='viridis')

        plt.title('Scatter Plot: Astronomy vs Defense Against the Dark Arts')
        plt.xlabel('Astronomy Score')
        plt.ylabel('Defense Against the Dark Arts Score')

        fig1 = plt.gcf()

        plt.show()

        fig1.savefig("results/scatter_plot.png")
        plt.close()
        print("Scatter plot saved to results/")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """
    Main function to execute the scatter plot generation.

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
        plot_scatter(dataset_path)
    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except pd.errors.ParserError:
        print("Error: The CSV file is malformed.")
    except Exception as e:
        print(f"An error occurred in the main function: {e}")


if __name__ == "__main__":
    main()
