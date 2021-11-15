import glob
import os


def read_paths(data_folder):
    """
    Read the file paths that are in the subfolders inside data_folder.

    :param data_folder: folder containg the subfolders with images
    :return: directory of classes with image paths inside the classes
    """
    # get the subfolders
    class_paths = glob.glob(os.path.join(data_folder, '*'))

    # get all the labels and the datapaths
    return {class_path.split('/')[-1]: read_image_paths_in_folder(class_path) for class_path in class_paths}


def read_image_paths_in_folder(folder):
    """
    Read the paths of all the images in a folder.
    Images should be .jpg or .png .

    :param folder: folder path to look for images
    :return: the paths of all the images
    """
    return glob.glob(os.path.join(folder, '*.jpg')) + glob.glob(os.path.join(folder, '*.png'))
