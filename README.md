# 🌳🌍 Tropical Forest Conservation via Spectral Band Selection using UMDA and DeepLabv3+ 🌍🌳

## Overview

This repository contains the code and data for the paper **"A Satellite Band Selection Framework for Amazon Forest Deforestation Detection Task"**.

The conservation of tropical forests is critical due to their significant ecological roles. This study introduces a novel framework employing the Univariate Marginal Distribution Algorithm (UMDA) to select optimal spectral bands from Landsat-8 satellite imagery, enhancing the performance of deforestation detection through a semantic segmentation architecture (DeepLabv3+).

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3918/)

## 🌿 Abstract 🌿

The conservation of tropical forests is a topic of significant social and ecological relevance due to their crucial role in the global ecosystem. Unfortunately, deforestation and degradation impact millions of hectares annually, necessitating government or private initiatives for effective forest monitoring. This study introduces a novel framework that employs the Univariate Marginal Distribution Algorithm (UMDA) to select spectral bands from Landsat-8 satellite imagery, optimizing the representation of deforested areas. This selection guides a semantic segmentation architecture, DeepLabv3+, enhancing its performance. Experimental results revealed several band compositions that achieved superior balanced accuracy compared to commonly adopted combinations for deforestation detection, utilizing segment classification via a Support Vector Machine (SVM). Moreover, the optimal band compositions identified by the UMDA-based approach improved the performance of the DeepLabv3+ architecture, surpassing state-of-the-art approaches compared in this study. The observation that a few selected bands outperform the total contradicts the data-driven paradigm prevalent in the deep learning field. Therefore, this suggests an exception to the conventional wisdom that 'more is always better'.

## 🌲 Table of Contents 🌲

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## 🍃 Introduction 🍃

The goal of this project is to improve the accuracy of deforestation detection by selecting the most relevant spectral bands from Landsat-8 satellite imagery. The Univariate Marginal Distribution Algorithm (UMDA) is used for this purpose, allowing the selection of the most informative bands for the task. The selected bands are then used to train a DeepLabv3+ model for semantic segmentation of deforested areas.

## 🌱 Installation 🌱

### Prerequisites

All dependencies are listed in the `requirements.txt` file. To install them, simply run:
    
```bash
pip install -r requirements.txt
```


### Clone the Repository

```bash
git clone https://github.com/ebouhid/BandSelection.git
cd BandSelection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🌳 Usage 🌳

### Data Preparation

1. 🌐 Download Landsat-8 satellite imagery from [Google Drive](https://drive.google.com/file/d/1EbrG_-khBsHx5y3eMfRaozYT_sI9EArb/view?usp=sharing). It contains all the files needed to replicate the experiments in our paper.
2. 📦 Extract the downloaded files to the `data/` directory.

### Prepare your own UMDA dataset
If you want to use a different train/validation/test split, you can create your own dataset by following these steps:
1. Change the regions specified in the `generate_bandstudy_segments.py` script to the desired split.
```python
val_region = ['x08']
test_regions = ['x03', 'x04']
```
2. Execute the `generate_classification_data.sh` script.
```bash
./generate_classification_data.sh
```

You can also use different satellite scenes, truth masks or superpixels by changing the paths in the `generate_bandstudy_segments.py` script 😉.


### Running the Experiments

1. **Band Selection using UMDA**:

To run the experiment with 5 different seeds in parallel (like we did in the paper), use the following command:
```bash
./run_5_seeds.sh your_experiment_name
```

To run a single experiment, use the following command:
```bash
python python umda.py --exp-name your_experiment_name --seed your_seed
```

2. **Parse UMDA results**:

Currently, the UMDA results are parsed manually. We are working on a script to automate this process.
The parsing process is done as specified in the paper (see Table 3/Section 4.1).
<!-- ```bash
python train_deeplabv3.py --config configs/train_config.yaml
``` -->

3. **Train (and test) the Semantic Segmentation model**:

The train/test process is pretty straightforward. The `train.py` script will train the model using the specified configurations (right now, the configurations are hardcoded in the script but they will be supplied as command line arguments soon).
```python
MODEL_NAME = 'DeepLabV3Plus' # Specify model architecture (see SegmentationModelsPyTorch documentation)
BATCH_SIZE = 16 # Specify batch size
NUM_EPOCHS = 100 # Set the number of epochs
PATCH_SIZE = 256 # Set the patch size
STRIDE_SIZE = 64 # Set the stride size for the train set sliding window
NUM_CLASSES = 1 # Set the number of classes (deforestation and non-deforestation)
DATASET_DIR = '../data/scenes_allbands_ndvi' # Set the path to the satellite images
GT_DIR = '../data/truth_masks' # Set the path to the ground truth masks
# Set the band composition (this example uses the bands 6, 5 and 1 from the Landsat-8 Operational Land Imager).
# Here, you can take the `n` most relevant bands from the UMDA results.
COMPOSITION = [6, 5, 1]
```

```bash
cd segmentation
python train.py
```


## 📧 Contact 📧

For any questions or inquiries, please contact:

- **Eduardo Bouhid** - ebneto *(at)* unifesp *(dot)* br | [LinkedIn](https://www.linkedin.com/in/eduardo-bouhid-neto-227a56b7/)
- **Fabio A. Faria** - ffaria *(at)* unifesp *(dot)* br | [LinkedIn](https://www.linkedin.com/in/fabio-augusto-faria-ba754724/)
