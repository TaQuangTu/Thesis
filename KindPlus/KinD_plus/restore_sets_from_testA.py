
import os
import shutil

test_folder = "test_results/"
def get_all_files(paths):
    items = os.listdir(test_folder)
    for item in items:
            paths.append(item)
if __name__ == "__main__":
    paths = []
    get_all_files(paths)
    print(len(paths))
    for path in paths:
        octets = path.split("_")
        folder = octets[0]+"/"+octets[1]+"/"+octets[2]
        if not os.path.exists(octets[0]):
            os.mkdir(octets[0])
        if not os.path.exists(octets[0]+"/"+octets[1]):
            os.mkdir(octets[0]+"/"+octets[1])
        if not os.path.exists(octets[0]+"/"+octets[1]+"/"+octets[2]):
            os.mkdir(octets[0]+"/"+octets[1]+"/"+octets[2])
        print(folder+"/"+octets[3])
       # shutil.copy(test_folder+path,folder+"/"+octets[3])
