import os
from os.path import join
import shutil

def get_visible_anno_file_paths(dir, result):
    a = os.listdir(dir)
    for item in a:
        path = join(dir, item)
        if os.path.isdir(path):
            get_visible_anno_file_paths(path, result)
        elif os.path.isfile(path):
            result.append(path)
def copy_to_test_set(in_files,to_folder):
    if not os.path.exists(to_folder):
        os.mkdir(to_folder)
    new_files = [x.replace("/","_") for x in in_files]
    for index, file in  enumerate(in_files):
        shutil.copy(file,to_folder+"/"+new_files[index])

if __name__=="__main__":
    in_folders = ["set09","set10","set11"]
    for folder in in_folders:
        print("copying__",folder)
        in_files = []
        get_visible_anno_file_paths(folder, in_files)
        copy_to_test_set(in_files, "test_images")

    

