# -*- coding: utf-8 -*-
import os
import zipfile


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(package):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    ensure_folder('data')

    extract('ai_challenger_pdr2018_trainingset_20180905')
    extract('ai_challenger_pdr2018_validationset_20180905')
    extract('ai_challenger_pdr2018_testA_20180905')
