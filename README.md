Dataset

https://www.kaggle.com/code/spaceman04/chest-x-rays/input


# ML in Cardiac Identification Using Bounding Boxes

Deep-learning localisation of the heart in chest X-ray images. Six ImageNet-pretrained CNN backbones (AlexNet, VGG16, GoogLeNet, ResNet-18, ResNet-50, EfficientNet) are fine-tuned to regress a bounding box (x_min, y_min, x_max, y_max) around the cardiac silhouette, and their performance is compared using bounding-box regression error and Intersection-over-Union (IoU).

This repository contains the code and analysis notebooks accompanying the publication:

> Narendra Rathod, Kriti Awasthi, Magdalena Kostkiewicz, Ewa Stępień.
> **Advancing Cardiac Detection in Chest X-ray Images Using Machine Learning: A Practical Application of AI in Medical Imaging.**
> *Bio-Algorithms and Med-Systems* 20(1), 2024.
> https://doi.org/10.5604/01.3001.0054.9269

## Funding

This work was supported by **Grant No. 2022/47/I/NZ7/03112** awarded by the **National Science Centre (NCN), Poland**.

## License

The code, notebooks and annotations in this repository are released under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
See [`LICENSE`](LICENSE) or https://creativecommons.org/licenses/by/4.0/.

You are free to share and adapt this material for any purpose, including commercially, provided you give appropriate credit (see [Citation](#citation)).

**Note on the image data:** the chest X-ray images themselves are *not* redistributed here. They originate from the RSNA Pneumonia Detection Challenge dataset hosted on Kaggle and remain subject to the terms of that dataset (see [Dataset](#dataset)). The CC BY 4.0 license covers only the material authored in this repository.

## Dataset

- **Source:** RSNA Pneumonia Detection Challenge chest radiographs (Kaggle).
  Current mirror used in this work: https://www.kaggle.com/code/spaceman04/chest-x-rays/input
  Kaggle occasionally relocates datasets; if the link is stale, search Kaggle for *"RSNA Pneumonia Detection Challenge"* or *"Chest X-rays"*.
- **Labels:** heart bounding boxes are read from `rsna_heart_detection.csv` with columns `name, x0, y0, w, h` (top-left corner plus width and height, in pixels of the preprocessed image).
- **Preprocessing:** DICOM images are converted to normalised `.npy` arrays and split into `Processed-Heart-Detection/train/` and `Processed-Heart-Detection/val/`. Subject lists are stored in `train_subjects.npy` and `val_subjects.npy`. Intensities are standardised with mean 0.494 and standard deviation 0.252.

The processed arrays and label CSV are not included in the repository; regenerate them with `Dataset_after_Preprocessing.ipynb` after downloading the raw data.

## Repository contents

| File | Purpose |
|---|---|
| `Dataset_after_Preprocessing.ipynb` | Defines `CardiacDataset` (PyTorch `Dataset`) that loads an image with its heart bounding box and applies joint image/box augmentation (gamma contrast, affine scale ±20 %, rotation ±10°, translation ±10 px) via `imgaug`. |
| `Training_final_AlexNet.ipynb` | Fine-tuning and evaluation with an AlexNet backbone. |
| `Training_final_VGG16.ipynb` | Fine-tuning and evaluation with a VGG16 backbone. |
| `Training_final_GoogleNet.ipynb` | Fine-tuning and evaluation with a GoogLeNet backbone. |
| `Training_final_RESNET18.ipynb` | Fine-tuning and evaluation with a ResNet-18 backbone. |
| `Training_final_RESNET50.ipynb` | Fine-tuning and evaluation with a ResNet-50 backbone. |
| `Training_final_EfficientNet.ipynb` | Fine-tuning and evaluation with an EfficientNet backbone. |
| `Plotting_histogram.ipynb` | Histograms of per-image bounding-box error for each backbone, used for the comparison figures in the paper. |

## Method

- **Task:** single-object regression. The final fully-connected layer of each backbone is replaced by a 4-unit linear layer; the first convolution is adapted to a single grayscale input channel.
- **Loss:** mean squared error on the four box coordinates.
- **Optimiser:** Adam, learning rate 1e-4.
- **Training:** PyTorch Lightning, up to 100 epochs, batch size 8, best checkpoint selected on validation loss.
- **Evaluation:** IoU between predicted and ground-truth boxes, plus qualitative overlays on validation images.

## Requirements

Python ≥ 3.8 with:

```
torch
torchvision
pytorch-lightning
torchmetrics
numpy
pandas
opencv-python
imgaug
matplotlib
```

Install with:

```bash
pip install torch torchvision pytorch-lightning torchmetrics numpy pandas opencv-python imgaug matplotlib
```

A CUDA-capable GPU is recommended; adjust `batch_size` and `num_workers` in the training notebooks to your hardware.

## How to reproduce

1. Download the chest X-ray data from Kaggle (see [Dataset](#dataset)) and obtain `rsna_heart_detection.csv`.
2. Run `Dataset_after_Preprocessing.ipynb` to create the `.npy` arrays, subject splits and to verify the augmentation pipeline.
3. Save the `CardiacDataset` class as `dataset.py` in the repository root (the training notebooks import it with `from dataset import CardiacDataset`).
4. Run any `Training_final_<backbone>.ipynb` notebook. Checkpoints and TensorBoard logs are written to `./logs/`.
5. Run `Plotting_histogram.ipynb` to reproduce the error-distribution comparison.

## Citation

If you use this code, please cite:

```bibtex
@article{Rathod2024CardiacDetection,
  author  = {Rathod, Narendra and Awasthi, Kriti and Kostkiewicz, Magdalena and St{\k{e}}pie{\'n}, Ewa},
  title   = {Advancing Cardiac Detection in Chest X-ray Images Using Machine Learning: A Practical Application of AI in Medical Imaging},
  journal = {Bio-Algorithms and Med-Systems},
  volume  = {20},
  number  = {1},
  year    = {2024},
  doi     = {10.5604/01.3001.0054.9269}
}
```

## Acknowledgements

Funded by the National Science Centre (NCN), Poland, Grant No. 2022/47/I/NZ7/03112. Chest radiographs courtesy of the Radiological Society of North America (RSNA) Pneumonia Detection Challenge.

## Contact

Narendra Rathod — narendra.rathod@unibe.ch
Kriti Awasthi - kriti.awasthi@doctoral.uj.edu.pl
