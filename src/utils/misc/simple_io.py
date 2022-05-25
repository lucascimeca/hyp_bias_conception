import shutil
import numpy as np
import os
import platform
from pathlib import Path
from os import path as pt


def file_exist_query(filename):
    path = Path(filename)
    if path.is_file():
        res = None
        while res not in ['y', 'Y', 'n', 'N']:
            res = input("\nThe file in '{}' already exists, do you really wish to re-write its contents? [y/n]".format(filename))
            if res not in ['y', 'Y', 'n', 'N']:
                print("Please reply with 'y' or 'n'")
        if res in ['n', 'N']:
            return False
    return True


def file_exists(filename):
    path = Path(filename)
    if path.is_file():
        return True
    return False


def remove_file(filename):
    try:
        os.remove(filename)
        return True
    except:
        return False


def folder_exists(folder_name):
    return pt.isdir(folder_name)


def folder_create(folder_name, exist_ok=False, parents=True):
    path = Path(folder_name)
    try:
        if exist_ok and folder_exists(folder_name):
            return True
        path.mkdir(parents=parents)
    except Exception as e:
        raise e
    return True


def remove_directory(path):
    if folder_exists(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            print("Error: %s : %s" % (path, e.strerror))
        return True
    else:
        return False


def get_filenames(folder="./", file_ending='', file_beginning='', contains=''):
    return list(np.sort([file for file in os.listdir(folder)
                         if file.endswith(file_ending)
                         and file.startswith(file_beginning)
                         and contains in file]))


def creation_date(path_to_file):
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            return stat.st_mtime
