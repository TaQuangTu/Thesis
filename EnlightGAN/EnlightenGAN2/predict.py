import os
from os.path import join
from random import random

from PIL import Image
import torchvision.transforms as transforms
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import torch
import cv2
import numpy as np

opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip


# ==============================================================================================
# data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
# model = create_model(opt)
#
# print(len(dataset))
# for i, data in enumerate(dataset):
#     path = data["A_paths"]
#     new_path = path.replace("/", "_")
#     octets = new_path.split("_")[-4:]
#     print("processed ",i,path)
#     model.set_input(data)
#     visuals = model.predict()
#
#     res = np.concatenate((visuals['fake_B'][:,:,2:3],visuals['fake_B'][:,:,1:2],visuals['fake_B'][:,:,0:1]),axis=2)
#
#     cv2.imwrite("TestResults/"+octets[0]+"/"+octets[1]+"/"+octets[2]+"/"+octets[3], res)
# =============================================================================================
def get_visible_anno_file_paths(dir, result):
    a = os.listdir(dir)
    for item in a:
        result.append(item)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        zoom = 1 + 0.1 * random.randint(0, 4)
        osize = [int(400 * zoom), int(600 * zoom)]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    # elif opt.resize_or_crop == 'no':
    #     osize = [384, 512]
    #     transform_list.append(transforms.Scale(osize, Image.BICUBIC))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

print("loading model=====================================")
model = create_model(opt)
print("loading test A")
images_paths = []
get_visible_anno_file_paths("test_dataset/testA", images_paths)
B_paths = []
get_visible_anno_file_paths("test_dataset/testB",B_paths)
B_imgs = [Image.open("test_dataset/testB/"+x).convert('RGB') for x in B_paths]
transform = get_transform(opt)
for i, path in enumerate(images_paths):
    img = Image.open("test_dataset/testA/"+path).convert('RGB')
    A_img = img
    A_path = path
    B_img = B_imgs[i % len(B_imgs)]
    B_path = B_paths[i % len(B_paths)]

    A_img = transform(A_img)
    B_img = transform(B_img)

    if opt.resize_or_crop == 'no':
        r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        A_gray = torch.unsqueeze(A_gray, 0)
        input_img = A_img
        # A_gray = (1./A_gray)/255
    else:
        w = A_img.size(2)
        h = A_img.size(1)

        # A_gray = (1./A_gray)/255.
        if (not opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A_img = A_img.index_select(2, idx)
            B_img = B_img.index_select(2, idx)
        if (not opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A_img = A_img.index_select(1, idx)
            B_img = B_img.index_select(1, idx)
        if opt.vary == 1 and (not opt.no_flip) and random.random() < 0.5:
            times = random.randint(opt.low_times, opt.high_times) / 100.
            input_img = (A_img + 1) / 2. / times
            input_img = input_img * 2 - 1
        else:
            input_img = A_img
        if opt.lighten:
            B_img = (B_img + 1) / 2.
            B_img = (B_img - torch.min(B_img)) / (torch.max(B_img) - torch.min(B_img))
            B_img = B_img * 2. - 1
        r, g, b = input_img[0] + 1, input_img[1] + 1, input_img[2] + 1
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        A_gray = torch.unsqueeze(A_gray, 0)
    input = {'A': A_img.unsqueeze(0), 'B': B_img.unsqueeze(0), 'A_gray': A_gray.unsqueeze(0),
             'input_img': input_img.unsqueeze(0),
             'A_paths': [A_path], 'B_paths': [B_path]}

    if not os.path.exists("TestResults"):
        os.mkdir("TestResults")
    
    print("TestResults/" + A_path)

    model.set_input(input)
    visuals = model.predict()

    res = np.concatenate((visuals['fake_B'][:, :, 2:3], visuals['fake_B'][:, :, 1:2], visuals['fake_B'][:, :, 0:1]),
                         axis=2)
    cv2.imwrite("TestResults/"+A_path, res)



