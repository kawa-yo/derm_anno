#!/usr/bin/env python3
from __future__ import annotations

from collections import OrderedDict
import time
import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v1

from typing import Any, List, Generator, Tuple
from nptyping import NDArray, Shape

from .io import make_parent_dir


class DermTiffImage:
    def __init__(self,
                 bg_image: NDArray[Shape["*, *, 3"], np.uint8],  # [H, W, C]
                 label2mask: OrderedDict[str,
                                         NDArray[Shape["*, *"], np.bool_]] = None,
                 label2color: OrderedDict[str,
                                          Tuple[np.uint8, np.uint8, np.uint8]] = None,
                 ) -> None:

        assert label2color.keys() == label2mask.keys()
        assert isinstance(bg_image, NDArray[Shape["*, *, 3"], np.uint8])
        for mask in label2mask.values():
            assert isinstance(mask, NDArray[Shape["*, *"], np.bool_])

        self.bg_image = bg_image

        self.label2mask = label2mask or OrderedDict()
        self.label2color = label2color or OrderedDict()

    @property
    def shape(self):
        return self.bg_image.shape

    @property
    def labels(self) -> List[str]:
        return list(self.label2mask.keys())

    def resize(self, width: int, height: int) -> DermTiffImage:
        bg_image = np.array(Image.fromarray(
            self.bg_image).resize((width, height)))
        masks = {
            label: np.array(Image.fromarray(mask).resize((width, height)))
            for label, mask in self.label2mask.items()
        }
        return DermTiffImage(bg_image, masks, self.label2color)

    # return (1-alpha) * bg_image
    #      + alpha * annotation
    def get_annotation_image(self,
                             label_list: List[str] = None,
                             alpha: float = 1.0,
                             ) -> Image.Image:

        if label_list is None:
            label_list = self.labels

        assert all([label in self.labels for label in label_list])
        assert 0 <= alpha <= 1

        annotated_image = np.copy(self.bg_image).astype(float)

        for label in label_list:
            mask = self.label2mask[label]
            color = np.array(self.label2color[label], dtype=float)
            annotated_image[mask] = (1.0 - alpha) * \
                annotated_image[mask] + alpha * color

        annotated_image = np.clip(
            annotated_image,
            a_min=0,
            a_max=255,
        ).astype(np.uint8)

        return Image.fromarray(annotated_image)

    def remove_frame(self, label: str) -> bool:

        if label not in self.label2mask:
            return False

        del self.label2mask[label]
        del self.label2color[label]
        return True

    def add_frame(self,
                  label: str,
                  mask: NDArray[Shape["*, *"], np.bool_],
                  color: Tuple[np.uint8, np.uint8, np.uint8]
                  ) -> bool:

        if label in self.label2mask:
            return False

        self.label2mask[label] = mask
        self.label2color[label] = color
        return True

    def save(self,
             output_file: str,
             compression: str = "tiff_adobe_deflate",
             mkdir: bool = True,
             verbose: bool = False,
             ) -> None:

        assert compression in ["tiff_lzw", "tiff_adobe_deflate"]

        _start_time = time.time()

        rgb_array = np.array(self.bg_image)
        H, W, _ = rgb_array.shape
        alpha = 255 * np.ones((H, W, 1), np.uint8)
        bg_image = Image.fromarray(np.concatenate([rgb_array, alpha], axis=2))

        frame_list = []
        for label in self.labels:
            color = self.label2color[label]
            mask = self.label2mask[label]

            R, G, B = color

            rgba = np.array((R, G, B, 255), np.uint8)
            rgba_array = np.zeros([H, W, 4], np.uint8)
            rgba_array[mask] = rgba

            # Image
            image = Image.fromarray(rgba_array)

            tiffinfo = ImageFileDirectory_v1()
            tiffinfo[285] = f'{label}/({R},{G},{B},255)'.encode("utf-8")
            image.tag = tiffinfo

            frame_list.append(image)

        if mkdir:
            make_parent_dir(output_file)

        bg_image.save(output_file,
                      append_images=frame_list,
                      save_all=True,
                      compression=compression,
                      )

        if verbose:
            _elapsed_time = time.time() - _start_time
            print('Elapsed time: {:.1f} [s]'.format(_elapsed_time))


def _tiffFrameGenerator(tiff_image: Image.Image,
                        ) -> Generator[Image.Image, None, None]:
    for i in range(tiff_image.n_frames):
        tiff_image.seek(i)
        yield tiff_image


def load_image(tiff_file: str,
               verbose: bool = False,
               ) -> DermTiffImage:

    label2mask = OrderedDict()
    label2color = OrderedDict()

    with Image.open(tiff_file) as tiff_image:
        for i, frame in enumerate(_tiffFrameGenerator(tiff_image)):
            frame = np.array(frame, np.uint8)
            if i == 0:
                frame = frame[:, :, :3]  # RGBA to RGB
                bg_image = frame
            else:
                assert 285 in tiff_image.tag
                # 285 : "LABEL/(R,G,B,A)"
                label, color = tiff_image.tag[285][0].split('/')
                color = color[1:-1].split(',')
                assert label not in label2mask, f"frame name duplicated: {label}."

                R, G, B, _ = map(int, color)
                label2mask[label] = frame[:, :, 3] != 0
                label2color[label] = (R, G, B)

                if verbose:
                    print(f'{i:2d}: Frame name = {label}, Color = {color}')

    return DermTiffImage(bg_image,
                         label2mask,
                         label2color)
