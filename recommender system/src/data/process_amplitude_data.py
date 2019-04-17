# this script is used for decompress the files downloaded from amplitude

import zipfile
import gzip
import os


def ungz(input_path, output_path):

    """
    Decompress the .gz files in a folder to json files.

    :param input_path: str, the path of folder of .gz files
    :param output_path: str, the path of decompressed files
    """

    if not os.path.isdir(input_path):
        raise ValueError('%s is not existed'%input_path)

    gz_files = os.listdir(input_path)
    for gz_file in gz_files:
        if gz_file[-2:] == 'gz':
            with gzip.open(input_path + gz_file, 'r') as json_file:
                json_data = json_file.read().decode('GBK')
                output_file = open(output_path + gz_file[:-3], 'w')
                output_file.write(json_data)
                output_file.close()


def unzip(input_path, output_path):

    """
    Extract the zip file of raw data from amplitude platform to a folder contains .gz of json files.

    :param input_path: str, the path of folder of .zip files
    :param output_path: str, the path of decompressed files
    """

    if not os.path.isdir(input_path):
        raise ValueError('%s is not existed!' % input_path)

    zip_files = os.listdir(input_path)
    for file in zip_files:
        if file[-3:] == 'zip':
            zip_file = zipfile.ZipFile(input_path + file, 'r')
            zip_file.extractall(members=zip_file.namelist(), path=input_path)
            ungz(input_path + '225588/', output_path)
            print('%s has been extracted.' % file)


if __name__ == "__main__":
    input_data_path = '../../data/raw/'
    output_data_path = '../../data/raw/raw_json/'
    unzip(input_data_path, output_data_path)
    print('Finish the current extracting process.')
