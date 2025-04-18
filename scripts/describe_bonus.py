# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#   describe_bonus.py                                  :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2025/02/19 22:40:35 by iberegsz          #+#    #+#              #
#   Updated: 2025/04/02 11:45:19 by iberegsz         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import csv
import sys


def calculate_statistics(data):
    """
    Calculate statistical metrics for numeric columns in the dataset."

    Args:
        data (list of dict): The dataset as a list of dictionaries, where each
                              dictionary represents a row.

    Raises:
        ValueError: If a non-numeric value is encountered in a numeric column.

    Returns:
        dict or None: A dictionary containing statistics (count, missing,
                      mean, std, var, min, max, range, 10%, 25%, 50%, 75%, 90%,
                      skewness, mode) for each numeric column,
                      or None if no statistics are calculated.
    """
    try:
        stats = {}
        for column in data[0].keys():
            if column in [
                    "Index",
                    "Hogwarts House",
                    "First Name",
                    "Last Name",
                    "Birthday",
                    "Best Hand"]:
                continue

            values = []
            missing = 0
            for row in data:
                if row[column].strip() == "":
                    missing += 1
                else:
                    values.append(float(row[column]))

            if not values:
                continue

            n = len(values)
            sorted_values = sorted(values)
            mean = sum(values) / n
            variance = sum((x - mean) ** 2 for x in values) / n
            std = variance ** 0.5

            if std == 0:
                skewness = 0.0
                kurtosis = 0.0
            else:
                skewness = (
                    sum((x - mean) ** 3 for x in values) / n) / (std ** 3)
                kurtosis = (sum((x - mean) ** 4 for x in values) / n) / \
                    (std ** 4) - 3

            percentiles = {
                '10%': sorted_values[int(n * 0.10)],
                '25%': sorted_values[int(n * 0.25)],
                '50%': sorted_values[int(n * 0.50)],
                '75%': sorted_values[int(n * 0.75)],
                '90%': sorted_values[int(n * 0.90)]
            }

            frequency = {}
            for x in values:
                frequency[x] = frequency.get(x, 0) + 1
            max_freq = 0
            mode = None
            for k, v in frequency.items():
                if v > max_freq:
                    max_freq = v
                    mode = k

            min_val = values[0]
            max_val = values[0]
            for v in values:
                if v < min_val:
                    min_val = v
                if v > max_val:
                    max_val = v

            stats[column] = {
                'count': n,
                'missing': missing,
                'mean': mean,
                'std': std,
                'var': variance,
                'min': min_val,
                'max': max_val,
                'range': max_val - min_val,
                '10%': percentiles['10%'],
                '25%': percentiles['25%'],
                '50%': percentiles['50%'],
                '75%': percentiles['75%'],
                '90%': percentiles['90%'],
                'skewness': skewness,
                'kurtosis': kurtosis,
                'mode': mode
            }
        return stats
    except ValueError as e:
        print(f"Error processing numeric values in calculate_statistics: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred in calculate_statistics: {e}")
        raise


def describe_dataset(filepath):
    """
    Describe the dataset by calculating and printing statistics for each"
    numeric column.

    Args:
        filepath (str): The path to the CSV file containing the dataset.

    Raises:
        ValueError: If a non-numeric value is encountered in a numeric column.
        FileNotFoundError: If the specified file does not exist.
        csv.Error: If there is an error reading the CSV file.
        Exception: If an unexpected error occurs during processing.

    Returns:
        None
    """
    try:
        with open(filepath, mode='r') as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]

        stats = calculate_statistics(data)

        column_mapping = {
            'Arithmancy': 'Arith.',
            'Astronomy': 'Astron.',
            'Herbology': 'Herb.',
            'Defense Against the Dark Arts': 'Defense',
            'Divination': 'Div.',
            'Muggle Studies': 'Muggle',
            'Ancient Runes': 'Ancient',
            'History of Magic': 'Hist. Mag.',
            'Transfiguration': 'Transf.',
            'Potions': 'Pot.',
            'Care of Magical Creatures': 'Care Creat.',
            'Charms': 'Charms',
            'Flying': 'Flying'
        }

        stats = {
            column_mapping.get(
                col,
                col): values for col,
            values in stats.items()}

        stat_params = [
            'count', 'missing', 'mean', 'std', 'var',
            'min', 'max', 'range', '10%', '25%', '50%', '75%', '90%',
            'skewness', 'kurtosis', 'mode'
        ]

        print(f"{'Feature':<12}", end="")
        for col in stats.keys():
            print(f"{col:>16}", end="")
        print()

        for param in stat_params:
            print(f"{param.capitalize():<12}", end="")
            for col in stats.values():
                value = col.get(param, None)
                if isinstance(value, float):
                    print(f"{value:>16.6f}", end="")
                else:
                    print(f"{value:>16}", end="")
            print()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        raise
    except csv.Error as e:
        print(f"Error reading CSV file in describe_dataset: {e}")
        raise
    except ValueError as e:
        print(f"Error processing numeric values in describe_dataset: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred in describe_dataset: {e}")
        raise


def main():
    """
    Main function to execute the script.
    It describes the dataset located at the default file path.

    Args:
        None

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        csv.Error: If there is an error reading the CSV file.
        ValueError: If a non-numeric value is encountered in a numeric column.
        Exception: If an unexpected error occurs during processing.

    Returns:
        None
    """
    try:
        if len(sys.argv) > 1:
            dataset_filepath = sys.argv[1]
        else:
            dataset_filepath = "data/dataset_train.csv"
        describe_dataset(dataset_filepath)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except csv.Error as e:
        print(f"Error: CSV reading issue - {e}")
    except ValueError as e:
        print(f"Error: Value error - {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        raise


if __name__ == "__main__":
    main()
