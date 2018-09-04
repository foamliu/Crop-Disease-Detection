import os
import zipfile
import shutil

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(package):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def flatten(folder):
    root = os.path.join('data', folder)
    folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    for folder in folders:
        path = os.path.join(root, folder)
        sub_folders = [sub for sub in os.listdir(path) if os.path.isdir(os.path.join(path, sub))]
        for sub in sub_folders:
            print(sub)


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    extract('ai_challenger_pdr2018_trainingset')
    extract('ai_challenger_pdr2018_validationset')
    extract('ai_challenger_pdr2018_testA')

    flatten('AgriculturalDisease_trainingset')
    flatten('AgriculturalDisease_validationset')