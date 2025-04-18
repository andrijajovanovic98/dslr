# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#   histogram.py                                       :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2025/02/18 11:16:30 by iberegsz          #+#    #+#              #
#   Updated: 2025/04/02 11:40:48 by iberegsz         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_histograms(filepath):
    """
    Generate histograms for specified courses in the dataset.

    Args:
        filepath (str): Path to the dataset CSV file.

    Raises:
        FileNotFoundError: If the dataset file is not found.
        Exception: If any other unexpected error occurs.

    Returns:
        None
    """
    try:
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        data = pd.read_csv(filepath)

        print("Columns in the dataset:", data.columns)
        print("First few rows of data:")
        print(data.head())

        courses = [
            'Astronomy',
            'Herbology',
            'Defense Against the Dark Arts',
            'Divination']

        for course in courses:
            if course not in data.columns:
                print(f"Column {course} not found in the dataset.")
                continue

            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(
                    data=data,
                    x=course,
                    hue='Hogwarts House',
                    kde=True,
                    palette='viridis')
                plt.title(f'Distribution of {course} Scores')
                plt.xlabel('Score')
                plt.ylabel('Frequency')

                fig1 = plt.gcf()

                plt.show()

                fig1.savefig(f"{results_dir}/{course}_histogram.png", dpi=100)
                plt.close()
                print(f"Histogram for {course} saved successfully.")

            except Exception as e:
                print(f"Error generating histogram for {course}: {e}")

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """
    Main function to execute the histogram plotting.
    It generates histograms for specified courses in the dataset.

    Args:
        None

    Raises:
        FileNotFoundError: If the dataset file is not found.
        Exception: If any other unexpected error occurs.

    Returns:
        None
    """
    try:
        dataset_path = sys.argv[1] if len(
            sys.argv) > 1 else "data/dataset_train.csv"
        plot_histograms(dataset_path)
    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except pd.errors.ParserError:
        print("Error: The CSV file could not be parsed.")
    except IndexError:
        print("Error: The dataset file path was not provided.")
    except ValueError:
        print("Error: The dataset file path is invalid.")
    except Exception as e:
        print(f"An error occurred in main(): {e}")


if __name__ == "__main__":
    main()
