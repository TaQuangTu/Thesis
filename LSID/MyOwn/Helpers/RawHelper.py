import rawpy
from PIL import Image
import numpy as np
class RawHelper:
    @staticmethod
    def raw_to_png(raw_path, png_path):
        raw_data = open(raw_path, 'rb').read()

        img = rawpy.imread(raw_path)
        width,height = img.raw_image.shape
        image_size = (width,height)
        img = Image.frombytes('L', image_size, raw_data)
        img.save(png_path)