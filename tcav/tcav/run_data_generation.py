import os

from .utils import download_concept_dataset


if __name__ == "__main__":
    download_concept_dataset(
        concepts=['laptop', 'bear', 'horse', 'knife', 'tie', 'train',
                  'zebra', 'dog'],
        num_imgs=50,
        random_state=42,
        path_save_dir=f"{os.path.abspath(os.path.dirname(__file__))}"
                      "/data/concept")

    for rs in range(50):
        os.makedirs(f"{os.path.abspath(os.path.dirname(__file__))}"
                    f"/data/random/random{rs}/random{rs}", exist_ok=True)

        download_concept_dataset(
            exclude_concepts=['laptop', 'bear', 'horse', 'knife', 'zebra',
                              'dog', 'tie', 'train'],
            num_imgs=50,
            random_state=rs,
            path_save_dir=f"{os.path.abspath(os.path.dirname(__file__))}"
                          f"/data/random/random{rs}/random{rs}")
