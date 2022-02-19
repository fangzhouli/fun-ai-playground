# -*- coding: utf-8 -*-
"""One line to describe the file.

This module...

Example:
    A paragraph to describe the output.

        $ python utils.py.py

Attributes:
    variable_1 (int): Description for module-level attribute.

Authors:
    Fangzhou Li: https://github.com/fangzhouli

TODO:
    * Finish comments.

"""
import os
import shutil
from typing import List

from PIL import Image
import numpy as np
from pycocotools.coco import COCO

PATH_MSCOCO_DIR = "/../../data/lfz/mscoco"


def download_concept_dataset(
        concepts: List[str],
        num_imgs: int = None,
        random_state: int = None,
        path_dir_save: str = None) -> None:
    """Load concept dataset from ImageNet dataset.

    Args:
        concepts: List of concepts.
        num_imgs: Number of images to load. Defaults to retrieve all.
        random_state: Random state. Defaults to None.
        path_dir_save: Path to save the dataset. Defaults to the working dir.

    Returns:
        None.

    """
    rng = np.random.default_rng(random_state)
    coco = COCO(f"{PATH_MSCOCO_DIR}/annotations/instances_train2017.json")

    for concept in concepts:
        ids_class = coco.getCatIds(catNms=concept)
        ids_img = coco.getImgIds(catIds=ids_class)
        if num_imgs is not None:
            ids_img = rng.choice(ids_img, num_imgs, replace=False)

        if path_dir_save is None:
            path_dir_save = "."
        os.mkdir(f"{path_dir_save}/{concept}")

        for id_img in ids_img:
            id_ann = coco.getAnnIds(
                imgIds=id_img, catIds=ids_class, iscrowd=None)[0]
            ann = coco.loadAnns(id_ann)[0]
            img_dict = coco.loadImgs([id_img])[0]

            bbox = ann['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            img = Image.open(
                f"{PATH_MSCOCO_DIR}/train2017/{img_dict['file_name']}")
            img = img.crop(bbox)
            img.save(f"{path_dir_save}/{concept}/{img_dict['file_name']}")


if __name__ == "__main__":
    download_concept_dataset(
        ['dog'],
        num_imgs=50,
        random_state=42,
        path_dir_save="data")
