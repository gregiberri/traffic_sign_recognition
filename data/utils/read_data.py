import glob
import os


def read_paths(data_folder):
    class_paths = glob.glob(os.path.join(data_folder, '*'))

    # get all the labels and the datapaths
    return {class_path.split('/')[-1]: read_image_paths_in_folder(class_path) for class_path in class_paths}


def read_image_paths_in_folder(folder):
    return glob.glob(os.path.join(folder, '*.jpg'))
