from kaggle.api.kaggle_api_extended import KaggleApi
from urllib.parse import urlsplit
import requests
import zipfile
import shutil
import gzip
import csv
import re
import os


def download_resource(url, filepath):
    """
    Download a resource from a given URL and save it to the specified file path.

    Parameters:
    - url (str): The URL of the resource to download.
    - filepath (str): The file path where the downloaded resource will be saved.

    Returns:
    - bool: True if the download was successful, False otherwise.
    """

    parent_folder = os.path.dirname(filepath)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        with open(filepath, "wb") as file:
            file.write(response.content)
        return True
    else:
        return False


def download_google_drive(url_or_file_id, filepath):
    """
    Download a file from Google Drive given its URL or file ID and save it to the specified file path.

    Parameters:
    - url_or_file_id (str): The URL or file ID of the file to download from Google Drive.
    - filepath (str): The file path where the downloaded file will be saved.

    Returns:
    - bool: True if the download was successful, False otherwise.
    """

    file_id = url_or_file_id
    if is_url(url_or_file_id):
        file_id = extract_google_drive_file_id(url_or_file_id)
    return download_google_drive(file_id, filepath)


def download_kaggle(dataset_name, download_dir):
    """
    Download a dataset from Kaggle given its name and save & extract it to the specified file path.

    Parameters:
    - dataset_name (str): The dataset name of the dataset to download from Google Drive.
    - download_dir (str): The file path where the downloaded file will be saved and extracted.

    Returns:
    - bool: True (TODO: fix this)
    """

    # Get the value of the HOME environment variable
    home_directory = os.getenv('HOME')

    kaggle_api_path = os.path.join(home_directory, '.kaggle/kaggle.json')

    if not os.path.exists(kaggle_api_path):
        raise RuntimeError("Unable to find Kaggle API key (missing '~/.kaggle/kaggle.json').")

    # Initialize Kaggle API
    api = KaggleApi()
    # Authenticate with your Kaggle credentials
    api.authenticate()

    # Create the directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Download the dataset
    api.dataset_download_files(dataset_name, path=download_dir, unzip=True)

    return True


def is_local_path(path):
    """
    Check if a given path is a local file path.

    Parameters:
    - path (str): The path to be checked.

    Returns:
    - bool: True if the path is a local file path, False otherwise.
    """
    scheme = urlsplit(path).scheme
    return scheme == '' or scheme.lower() in {'file', 'localhost'}


def is_url(string):
    """
    Check if a given string is a URL.

    Parameters:
    - string (str): The string to be checked.

    Returns:
    - bool: True if the string is a URL, False if it's not.
    """
    scheme = urlsplit(string).scheme
    return scheme.lower() not in {'', 'file'}


def extract_google_drive_file_id(url):
    """
    Extract the file ID from a Google Drive URL.

    Parameters:
    - url (str): The Google Drive URL.

    Returns:
    - str: The file ID extracted from the URL, or None if no file ID is found.
    """

    pattern = r"/file/d/([-\w]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None


def extract_zip(zip_path, directory):
    """
    Extract the zip file in the specified directory.

    Parameters:
    - zip_path (str): Path of the zip file to extract.
    - directory (str): Path where to extract the zip file.

    Returns:
    - bool: True (TODO: fix this)
    """

    # Create the extraction directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Open the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Extract all contents to the specified directory
        zip_ref.extractall(directory)

    return True


def read_tsv(file_path, skip_header=False, encoding='utf-8'):
    """
    Reads a TSV (Tab-Separated Values) file and returns a 2D array.

    Parameters:
    - file_path (str): The path to the TSV file.

    Returns:
    - list of lists: A 2D array with as many rows as the lines in the file
                    and as many columns as the elements in each row.
    """

    data = []
    with open(file_path, 'r', newline='', encoding=encoding) as file:
        tsv_reader = csv.reader(file, delimiter='\t')

        if skip_header:
            next(tsv_reader)

        for row in tsv_reader:
            data.append(row)

    return data


def extract_gz(input_file, output_file):
    """
    Extracts a gzipped file into a specified directory.

    Parameters:
    - gz_file_path (str): Path to the gzipped file.
    - output_directory (str): Directory to extract the contents.

    Returns:
    - bool: True if the input file exist, false otherwise (TODO: fix this)
    """

    # Ensure the input file exists
    if os.path.exists(input_file):

        # Extract the gz file into the output directory
        with gzip.open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        return True

    return False
