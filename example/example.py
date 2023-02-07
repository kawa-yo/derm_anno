from derm_tiff import load_image
from PIL import Image
import os
import numpy as np


def main():
    # DermAnnotationで作成されたTIFF画像を開く
    derm_image = load_image("example/imgs/input.tiff")

    # TIFF画像の各フレームの名前を取得
    labels = derm_image.labels
    print(labels)  # ['Pencil1', 'Pencil2', 'Pencil3']
    print(derm_image.label2color)

    # TIFF画像の各フレームのマスクを合成して出力
    for label in labels:
        img = derm_image.get_annotation_image([label], alpha=0.5)
        img.save(f"example/imgs/with_{label}.png")

    # どのフレームのマスクとも重複しないようなマスクを新たに追加する
    H, W, _ = derm_image.shape
    color = (255, 255, 0)
    new_mask = np.ones((H, W), dtype=np.bool_)
    for label in labels:
        mask = derm_image.label2mask[label]
        new_mask = np.logical_and(new_mask, np.logical_not(mask))

    # ラベルがすでにある場合は追加されない
    derm_image.remove_frame("empty_space")
    derm_image.add_frame("empty_space", new_mask, color)

    img = derm_image.get_annotation_image(["empty_space"], alpha=0.5)
    img.save("example/imgs/with_empty_space.png")

    # DermAnnotationで開けるTIFF形式で保存
    derm_image.save("example/imgs/output.tiff")

    # 全体でリサイズも可能
    resized = derm_image.resize(300, 200)


if __name__ == "__main__":
    main()
