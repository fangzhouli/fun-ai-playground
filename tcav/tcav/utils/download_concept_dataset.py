# -*- coding: utf-8 -*-
"""One line to describe the file.

This module...

Example:
    A paragraph to describe the output.

        $ python utils.py

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
        concepts: List[str] = None,
        exclude_concepts: List[str] = None,
        num_imgs: int = None,
        random_state: int = None,
        path_save_dir: str = None) -> None:
    """Load concept dataset from ImageNet dataset.

    Args:
        concepts: List of concepts.
        exclude_concepts: List of concepts to exclude.
        num_imgs: Number of images to load. Defaults to retrieve all.
        random_state: Random state. Defaults to None.
        path_save_dir: Path to save the dataset. Defaults to the working dir.

    Returns:
        None.

    """
    if concepts is not None and exclude_concepts is not None:
        raise ValueError("Only one of concepts and exclude_concepts can be "
                         "specified.")

    rng = np.random.default_rng(random_state)
    coco = COCO(f"{PATH_MSCOCO_DIR}/annotations/instances_train2017.json")

    if concepts is not None:
        for concept in concepts:
            ids_class = coco.getCatIds(catNms=concept)
            ids_img = coco.getImgIds(catIds=ids_class)
            if num_imgs is not None:
                ids_img = rng.choice(ids_img, num_imgs, replace=False)

            if path_save_dir is None:
                path_save_dir = "."
            os.makedirs(f"{path_save_dir}/{concept}/{concept}", exist_ok=True)

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
                img.save(
                    f"{path_save_dir}/{concept}/{concept}/"
                    f"{img_dict['file_name']}")
    else:
        ids_class_all = set(coco.getCatIds())
        ids_class_exclude = set(coco.getCatIds(catNms=exclude_concepts))
        ids_class = list(ids_class_all.difference(ids_class_exclude))

        ids_img = []
        for id_class in ids_class:
            ids_img += coco.getImgIds(catIds=[id_class])

        if num_imgs is not None:
            ids_img = rng.choice(ids_img, num_imgs, replace=False)

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
            img.save(
                f"{path_save_dir}/"
                f"{img_dict['file_name']}")
