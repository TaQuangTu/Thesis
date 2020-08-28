import os

def get_all_files(dir, files):
    items = os.listdir(dir)
    for item in items:
        if os.path.isdir(dir+"/"+item):
            get_all_files(dir+"/"+item,files)
        else:
            files.append(item)

file = open("pedestrian_exdark.txt","r")
lines = file.readlines()
lines = [line.replace("\n","") for line in lines]
kind_files = []
files = get_all_files("test_results",kind_files)
print(lines[10:20])
print(kind_files[10:20])
for f in kind_files:
    if not f in lines:
        os.remove("test_results/"+f)
