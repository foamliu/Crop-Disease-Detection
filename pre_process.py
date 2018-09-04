import os
import shutil
import zipfile


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(package):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def flatten(folder):
    num_folders = 0
    num_images = 0
    root = os.path.join('data', folder)
    folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    for folder in folders:
        parent = os.path.join(root, folder)
        sub_folders = [sub for sub in os.listdir(parent) if os.path.isdir(os.path.join(parent, sub))]
        for sub in sub_folders:
            src_path = os.path.join(parent, sub)
            dst_path = os.path.join(root, folder + sub)
            print(src_path + ' -> ' + dst_path)
            shutil.move(src_path, dst_path)
            num_folders += 1
            num_images += len([f for f in os.listdir(dst_path) if f.lower().endswith('.jpg')])

        if len(sub_folders) == 0:
            num_folders += 1
            num_images += len([f for f in os.listdir(parent) if f.lower().endswith('.jpg')])
        else:
            shutil.rmtree(parent)

    print('num_folders: ' + str(num_folders))
    print('num_images: ' + str(num_images))


if __name__ == '__main__':
    ensure_folder('data')

    extract('ai_challenger_pdr2018_trainingset')
    extract('ai_challenger_pdr2018_validationset')
    extract('ai_challenger_pdr2018_testA')

    flatten('AgriculturalDisease_trainingset')
    flatten('AgriculturalDisease_validationset')
