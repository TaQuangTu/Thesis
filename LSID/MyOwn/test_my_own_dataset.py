import os
import os.path
from os.path import join
import Helpers.ModelHelper as sid
from os.path import basename

def get_image_paths(dir, result):
    a = os.listdir(dir)
    for item in a:
        path = join(dir, item)
        if os.path.isdir(path):
            get_image_paths(path, result)
        elif os.path.isfile(path) and "txt" not in path and "py" not in path:
            result.append(path)


if __name__ == "__main__":
    # get all anno files
    result = []
    get_image_paths('TestImages/', result)
    checkpoint_dir = '../checkpoint/Sony/'
    result_dir = 'TestResults/'
    sid_model = sid.SIDModel(checkpoint_dir)
    print(result)
    print(len(result),"=================================================")
    for input_image_path in result:
        print("processing",input_image_path)
        octets = input_image_path.split("/")
        if not os.path.exists(result_dir+octets[1]):
            os.mkdir(result_dir+octets[1])
        save_path = result_dir+octets[1]+"/"+octets[2]
        print("save to",save_path)
        sid_model.run_png(input_image_path, save_path)
