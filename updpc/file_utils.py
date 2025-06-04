import json
import os
from os import listdir
from os.path import abspath, basename, dirname, exists, isdir, isfile, join, splitext

from natsort import natsorted


def makedirs(path, exist_ok=True, **kwargs):
    """
    Create a directory and its parents.
    By default, the function does not raise an error if the directory already exists.

    Parameters
    ----------
    path : str
        The path to the directory to create.
    exist_ok : bool, optional
        Whether to raise an error if the directory already exists. Default is True.
    """
    os.makedirs(path, exist_ok=exist_ok, **kwargs)


def filter_str_list(
    str_list, include=None, exclude=None, include_logic="any", exclude_logic="all"
):
    """
    Filter a list of strings based on inclusion and exclusion criteria.

    Parameters
    ----------
    str_list : list of str
        The list of strings to filter.
    include : str or list of str, optional
        The string or list of strings to include. Default is None.
    exclude : str or list of str, optional
        The string or list of strings to exclude. Default is None.
    include_logic : str, optional
        The logic to use for including strings. Options are "any" or "all". Default is "any".
    exclude_logic : str, optional
        The logic to use for excluding strings. Options are "any" or "all". Default is "all".

    Returns
    -------
    list of str
        The filtered list of strings.
    """
    if include is None and exclude is None:
        return str_list
    if include is not None:
        include = [include] if isinstance(include, str) else include
        if include_logic == "any" or "or":
            str_list = [s for s in str_list if any(inc in s for inc in include)]
        elif include_logic == "all" or "and":
            str_list = [s for s in str_list if all(inc in s for inc in include)]
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude
        if exclude_logic == "any" or "or":
            str_list = [s for s in str_list if not any(exc in s for exc in exclude)]
        elif exclude_logic == "all" or "and":
            str_list = [s for s in str_list if all(exc not in s for exc in exclude)]
    return str_list


def list_ext_files(dir_path, ext=None, include=None, exclude=None, **kwargs):
    """
    List files with a specific extension in a directory.
    Optionally, include or exclude files based on their names.
    The files are sorted in natural order.

    Parameters
    ----------
    dir_path : str
        The path to the directory to list files from.
    ext : str, optional
        The extension of the files to list. Default is None.
    include : str or list of str, optional
        The string or list of strings to include in the file names. Default is None.
    exclude : str or list of str, optional
        The string or list of strings to exclude from the file names. Default is None.

    Returns
    -------
    list of str
        The list of file paths. The paths are sorted in natural order.
    """
    if not isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if ext is not None:
        ext = ext if ext.startswith(".") else "." + ext
        file_list = [f for f in listdir(dir_path) if splitext(f)[1] == ext]
    file_list = natsorted(file_list)
    file_list = filter_str_list(file_list, include=include, exclude=exclude, **kwargs)
    return [join(dir_path, f) for f in file_list]


def list_csvs(dir_path, include=None, exclude=None, **kwargs):
    """
    List CSV files in a directory.
    Optionally, include or exclude files based on their names.
    The files are sorted in natural order.

    Parameters
    ----------
    dir_path : str
        The path to the directory to list CSV files from.
    include : str or list of str, optional
        The string or list of strings to include in the file names. Default is None.
    exclude : str or list of str, optional
        The string or list of strings to exclude from the file names. Default is None.

    Returns
    -------
    list of str
        The list of CSV file paths. The paths are sorted in natural order.
    """
    return list_ext_files(
        dir_path, ext="csv", include=include, exclude=exclude, **kwargs
    )


def list_tifs(dir_path, include=None, exclude=None, **kwargs):
    """
    List TIFF files in a directory.
    Optionally, include or exclude files based on their names.
    The files are sorted in natural order.

    Parameters
    ----------
    dir_path : str
        The path to the directory to list TIFF files from.
    include : str or list of str, optional
        The string or list of strings to include in the file names. Default is None.
    exclude : str or list of str, optional
        The string or list of strings to exclude from the file names. Default is None.

    Returns
    -------
    list of str
        The list of TIFF file paths. The paths are sorted in natural order.
    """
    return list_ext_files(
        dir_path, ext="tif", include=include, exclude=exclude, **kwargs
    )


def list_folders(dir_path):
    """
    List folders in a directory.
    The folders are sorted in natural order.

    Parameters
    ----------
    dir_path : str
        The path to the directory to list folders from.

    Returns
    -------
    list of str
        The list of folder paths. The paths are sorted in natural order.
    """
    folder_list = [f for f in listdir(dir_path) if isdir(join(dir_path, f))]
    folder_list = natsorted(folder_list)
    return [join(dir_path, f) for f in folder_list]


def print_list(lst):
    """
    Print a list of elements with their indices.

    Parameters
    ----------
    lst : list
        The list of elements to print
    """
    for i, elem in enumerate(lst):
        print(i, elem)


def line_breaks(n=1):
    """
    Print line breaks.

    Parameters
    ----------
    n : int, optional
        The number of line breaks. Default is 1.
    """
    for _ in range(n):
        print()


def basename_noext(path):
    """
    Get the basename of a path without the extension.

    Parameters
    ----------
    path : str
        The path to get the basename from.

    Returns
    -------
    str
        The basename without the extension.
    """
    return splitext(basename(path))[0]


def commonhead(str1, str2):
    """
    Get the common head of two strings.

    Parameters
    ----------
    str1 : str
        The first string.
    str2 : str
        The second string.

    Returns
    -------
    str
        The common head of the two strings.
    """
    ret = ""
    for s1, s2 in zip(str1, str2):
        if s1 == s2:
            ret += s1
        else:
            break
    return ret


def save_dict_to_json(data: dict, filename: str, exist_ok: bool = True) -> bool:
    """
    Saves a dictionary to a file in JSON format, with an option to prevent overwriting.

    Parameters
    ----------
    data : dict
        The dictionary to save.
    filename : str
        The name of the file where the dictionary will be saved.
    exist_ok : bool, optional
        If True (default), overwrites the file if it already exists.
        If False, raises a FileExistsError if the file already exists.

    Returns
    -------
    bool
        True if the operation was successful, False otherwise.
    """
    try:
        if not exist_ok and os.path.exists(filename):
            raise FileExistsError(
                f"The file '{filename}' already exists and overwriting is not allowed."
            )

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)  # Save in a readable format
        return True
    except Exception as e:
        print(f"Error saving dictionary to JSON: {e}")
        return False


def load_dict_from_json(filename: str) -> dict:
    """
    Loads a dictionary from a JSON file.

    Parameters
    ----------
    filename : str
        The name of the JSON file to load.

    Returns
    -------
    dict
        The dictionary loaded from the JSON file. If the file cannot be read,
        an empty dictionary is returned.
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading dictionary from JSON: {e}")
        return {}
