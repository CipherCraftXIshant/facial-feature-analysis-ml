# Raw Dataset

This folder is intended to hold the raw UTKFace dataset images.

Due to file size, the raw dataset is NOT tracked in this repository (see .gitignore). To reproduce this project:

1. Download the UTKFace dataset (aligned & cropped version) from its official source or Kaggle.
2. Place the extracted "UTKFace" folder containing the .jpg images inside this "data/raw/" directory, or update the DATASET_PATH / IMAGE_DIR variables in the notebooks to point to wherever you've stored it locally.
3. Run the notebooks in numerical order (01 through 11) to reproduce the full pipeline from raw images to the final preprocessed dataset.
