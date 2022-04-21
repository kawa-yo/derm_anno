#!/usr/bin/env python3


from collections import OrderedDict
import os
import time

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from PIL.TiffTags import TAGS
from PIL.TiffImagePlugin import ImageFileDirectory_v1

from typing import List, Generator


def _mkdir(dirname: str) -> None:
    if dirname.strip() != '':
        os.makedirs(dirname, exist_ok=True)


def _tiffFrameGenerator(tiff_image: Image.Image) -> Generator[Image.Image, None, None]:
    for i in range(tiff_image.n_frames):
        tiff_image.seek(i)
        yield tiff_image


class DermAnnoImage:
    def __init__(self,
                 bg_image: npt.ArrayLike,  # [H, W, C]
                 map_dict: OrderedDict[str, npt.ArrayLike],
                 color_dict: OrderedDict[str, List[int]],
                 ) -> None:

        self.bg_image = bg_image

        if map_dict is not None:
            self.map_dict = map_dict
            self.color_dict = color_dict
        else:
            self.map_dict = OrderedDict()
            self.color_dict = OrderedDict()

    def get_annotation_image(self,
                             layer_name_list: List[str] = None,
                             alpha: float = 1.0,
                             ) -> npt.ArrayLike:

        annotation_image = np.zeros(self.bg_image.shape, np.uint8)
        annotation_map = np.zeros(self.bg_image.shape[:2], np.uint8)

        if layer_name_list is None:
            layer_name_list = list(self.map_dict.keys())

        for layer_name in layer_name_list:
            if layer_name not in self.map_dict:
                continue
            index_set = np.where(self.map_dict[layer_name] == 1)
            color = self.color_dict[layer_name]
            annotation_image[index_set] = np.array(color, np.uint8)
            annotation_map[index_set] = 1
        index_set = np.where(annotation_map == 1)
        output_annotation_image = np.copy(self.bg_image)
        output_annotation_image[index_set] = np.clip(
            alpha * annotation_image[index_set] + (1.0 - alpha) * self.bg_image[index_set],
            0,
            255,
            ).astype(np.uint8)

        return output_annotation_image

    def add_layer(self,
                  layer_name: str,
                  color: List[int],
                  ) -> bool:
        if layer_name not in self.map_dict:
            self.map_dict[layer_name] = \
                    np.zeros(self.bg_image.shape[:2], np.uint8)
            self.color_dict[layer_name] = color
            return True
        return False

    def save(self,
             output_file: str,
             verbose: bool = False,
             remove_uncompressed_output_file: bool = True,
             ) -> None:

        start_time = time.time()
        image_list = []
        height, width = self.bg_image.shape[:2]
        array = cv2.cvtColor(self.bg_image, cv2.COLOR_BGR2RGBA)
        array[:, :, 3] = 255
        image = Image.fromarray(array)
        image_list.append(image)

        for layer_name, map_dict in self.map_dict.items():
            height, width = map_dict.shape[:2]
            color = self.color_dict[layer_name]
            array = np.full([height, width, 3], 255, np.uint8)
            array[map_dict == 1] = np.array(color, np.uint8)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGBA)
            array[:, :, 3] = (map_dict * 255).astype(np.uint8)
            image = Image.fromarray(array)
            B, G, R = color 

            tiffinfo = ImageFileDirectory_v1()
            tiffinfo[285] = f'{layer_name}/({R}, {G}, {B}, 255)'.encode("utf-8")
            image.tag = tiffinfo

            image_list.append(image)

        _mkdir(os.path.dirname(output_file))
        image_list[0].save(output_file,
                           append_images=image_list[1:],
                           save_all=True,
                           compression="tiff_adobe_deflate",
                           )

        if verbose:
            elapsed_time = time.time() - start_time
            print('Elapsed time: {:.1f} [s]'.format(elapsed_time))


def load_image(tiff_file: str,
               verbose: bool = False,
               ) -> DermAnnoImage:

    map_dict = OrderedDict()
    color_dict = OrderedDict()

    with Image.open(tiff_file) as tiff_image:
        for i, image in enumerate(_tiffFrameGenerator(tiff_image)):

            image = np.array(image, np.uint8)
            if i == 0:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                bg_image = image
            else:
                assert 285 in tiff_image.tag
                ## 285 : "LAYER_NAME/(R, G, B, A)"
                layer_name, color = tiff_image.tag[285][0].split('/')
                color = color[1:-1].split(', ')
                assert layer_name not in map_dict, f"layer name duplicated: {layer_name}."
                if verbose:
                    print(f'{i:2d}: Layer name = {layer_name}, Color = {color}')
                mask = (image[:, :, 3] != 0).astype(np.uint8)
                map_dict[layer_name] = mask
                BGR_color = list(map(int, [color[2], color[1], color[0]]))
                color_dict[layer_name] = BGR_color

    return DermAnnoImage(bg_image,
                         map_dict,
                         color_dict)
