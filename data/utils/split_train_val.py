import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split

from data.utils.read_data import read_paths


def get_paths_and_labels(data_dir):
    """
    Getting the data paths and corresponding labels and
    save the classnumber-classname file into classname_classnumber.csv.

    :param data_dir: the directory of the data: the csv files will be saved here
    :return: paths, labels
    """
    data_dict = read_paths(data_dir)
    names_numbers = dict(zip(data_dict.keys(), list(range(len(data_dict.keys())))))
    # split the data_dict and get lists of the paths and the corresponding label numbers
    paths, labels = zip(*[[value, names_numbers[key]] for key, value_list in data_dict.items() for value in value_list])
    assert len(paths) and len(labels) and len(paths) == len(labels), \
        f'Something went wrong with the read data paths and labels ({data_dir})'
    save_csv([names_numbers.values(), names_numbers.keys()], 'classnumber_classname.csv')

    return paths, labels


def split_train_val(split_rates, data_dir):
    """
    Split the data in the data dict according to the split rates and save the splits into train/val/test.csv files into the data_dir
    Also saves the classnumber-classname file into classname_classnumber.csv.

    :param split_rates: the percentage of the whole data for the train/val/test split (should add up to 1.0)
    :param data_dir: the directory of the data: the csv files will be saved here
    :return: (x_train, x_val, x_test, y_train, y_val, y_test) splitted datas and labels
    """
    assert sum(split_rates) == 1.0, 'The split rates should add up to 1.0'
    assert os.path.exists(data_dir), f'The data dir ({data_dir}) does not exist.'

    # read the data and save the classnumber-classname relations csv
    data, labels = get_paths_and_labels(data_dir)

    # make the splits and stratify to make sure that all the classes are represented in the val and test
    x_train, x_val, y_train, y_val = train_test_split(data, labels, train_size=split_rates[0],
                                                      stratify=labels, shuffle=True, random_state=0)

    # save the splits to csv files
    save_csv((x_train, y_train), os.path.join(data_dir, 'train.csv'))
    save_csv((x_val, y_val), os.path.join(data_dir, 'val.csv'))


def save_csv(columns, filename):
    """
    Save the columns to csv files in the dir under filename.csv

    :param columns: list containing the data in the columns of the csv file
    :param filename: the path for the csv file to be saved
    :return: None
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(zip(*columns))


def load_csv(filename):
    """
    Load the columns of a csv files in the dir under filename.csv

    :param filename: the path for the csv file to be load
    :return: the data inside the csv
    """
    with open(filename, "r", newline="") as f:
        csv_input = csv.reader(f)
        data = [row for row in csv_input]
    return data
