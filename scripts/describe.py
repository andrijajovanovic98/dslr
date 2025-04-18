# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#   describe.py                                        :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2025/02/19 09:54:31 by iberegsz          #+#    #+#              #
#   Updated: 2025/04/02 11:45:04 by iberegsz         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import csv
import sys


def calculate_statistics(data):
    """
    Calculate statistical metrics for numeric columns in the dataset.

    Args:
        data (list of dict): The dataset as a list of dictionaries, where each
                             dictionary represents a row.

    Raises:
        ValueError: If a non-numeric value is encountered in a numeric column.

    Returns:
        dict or None: A dictionary containing statistics (count, mean, std,
                      min, 25%, 50%, 75%, max) for each numeric column,
                      or None if no statistics are calculated.
    """
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

        values = [float(row[column]) for row in data if row[column] != ""]
        if not values:
            continue

        n = len(values)
        mean = sum(values) / n
        std = (sum((x - mean) ** 2 for x in values) / n) ** 0.5

        min_val = values[0]
        max_val = values[0]
        for v in values:
            if v < min_val:
                min_val = v
            if v > max_val:
                max_val = v

        sorted_values = sorted(values)

        stats[column] = {
            'count': n,
            'mean': mean,
            'std': std,
            'min': min_val,
            '25%': sorted_values[int(n * 0.25)],
            '50%': sorted_values[int(n * 0.50)],
            '75%': sorted_values[int(n * 0.75)],
            'max': max_val
        }
    return stats


def describe_dataset(filepath):
    """
    Read a dataset from a CSV file and print descriptive statistics
    for numeric columns.

    Args:
        filepath (str): The path to the CSV file containing the dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        csv.Error: If there is an error reading the CSV file.
        ValueError: If a non-numeric value is encountered in a numeric column.

    Returns:
    None
    """
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]

    stats = calculate_statistics(data)

    column_mapping = {
        'Arithmancy': 'Arithmancy',
        'Astronomy': 'Astronomy',
        'Herbology': 'Herbology',
        'Defense Against the Dark Arts': 'Defense D.A.',
        'Divination': 'Divination',
        'Muggle Studies': 'Muggle',
        'Ancient Runes': 'Ancient Run.',
        'History of Magic': 'Hist. Mag.',
        'Transfiguration': 'Transfig.',
        'Potions': 'Potions',
        'Care of Magical Creatures': 'Care Mg.Cr.',
        'Charms': 'Charms',
        'Flying': 'Flying'
    }

    stats = {
        column_mapping.get(
            col,
            col): values for col,
        values in stats.items()}

    column_width = 14
    param_column_width = 8
    print(f"{'Feature':<{param_column_width}}", end=" ")
    print("  ".join([f"{col:>{column_width}}" for col in stats.keys()]))

    stat_params = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    for param in stat_params:
        print(f"{param.capitalize():<{param_column_width}}", end=" ")
        print("  ".join(
            [f"{values[param]:>14.6f}" for col, values in stats.items()]))


def main():
    """
    Main function to execute the script. It describes the dataset located at
    the default file path.

    Args:
        None

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        csv.Error: If there is an error reading the CSV file.
        ValueError: If a non-numeric value is encountered in a numeric column.

    Returns:
        None
    """
    try:
        dataset_filepath = sys.argv[1] if len(
            sys.argv) > 1 else "data/dataset_train.csv"
        describe_dataset(dataset_filepath)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except csv.Error as e:
        print(f"Error: CSV reading issue - {e}")
    except ValueError as e:
        print(f"Error: Value error - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
