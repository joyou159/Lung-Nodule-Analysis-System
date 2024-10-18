# Lung Nodule Analysis System 


<p align="center">
  <img src="READMD-Assets\Project_Steps.png" alt="The project outlines" title="The project outlines" width="700" />
</p>

This project is an end-to-end deep learning pipeline for lung cancer detection using 3D CT scan data. It begins by identifying potential nodules in the lungs through segmentation, followed by filtering these candidates using a cascade of specialized CNN classifiers to assess malignancy.

## Lung Cancer Detection Pipline

### 1. Loading and processing raw data files 

The first step involves loading the raw **3D CT** scan data and preprocessing it to make it suitable for subsequent steps, which includes:  
- Transforming from **XYZ** (continuous) coordinates to **IRC** (discrete) coordinates, using the transformation directions and voxel dimensions attached in the metadata files. 

- Unifying the **XYZ** coordinates of the candidate nodules and annotations (actual nodules info). 

- Standardizing the voxel intensity (density values) to fit in the context of our problem. 

Relavant files: 

- [`utils/common_utils/util.py`](utils/common_utils/util.py): Contains general utility functions for data transformations.
- [`utils/CT.py`](utils/CT.py): Handles loading and preprocessing of CT scan data.

### 2. Segmentation 
Utilize a U-Net architecture to segment the lung region and detect candidate areas that potentially look like nodules. This step generates a heatmap of regions of interest, allowing the pipeline to focus on candidate nodules while discarding irrelevant anatomy. All the components of such subsystem are implemented from scratch, including:

- U-Net model architecture.
- Custom dataset with dataset balancing, GPU-based augmentation, and disk pre-caching for accelerated training.
- Fully fledged command-line application for training. It will
 parse command-line arguments, have a full-featured `--help` command, and be easy to run in a wide variety of environments.

Relevant files:

- [`utils/Unet.py`](utils/Unet.py): Implements the U-Net architecture used for segmentation.
- [`utils/seg_dset.py`](utils/seg_dset.py): Defines the custom dataset class for segmentation, including data loading and balancing.

- [`utils/seg_dset_utils.py`](utils/seg_dset_utils.py): Provides utilities for dataset augmentation.
- [`utils/seg_precaching.py`](utils/seg_precaching.py): Implements disk pre-caching functionalities to speed up the training process.

- [`seg_training.py`](seg_training.py): Implements command-line applicaiton for segmentation model training.

### 3. Nodule Grouping

Group the detected regions from the segmentation heatmap into clusters that represent potential nodules. Each cluster is identified by the coordinates of its center, providing a targeted approach for further analysis. This step acts as the bridge between the segmentation model and the CNN classifiers. 

To achieve this, we employ a simple connected-components algorithm for grouping suspected nodule voxels into chunks, which will then be passed to the classification step.

Relevant file:
- [**`nodule_analysis_pipline.py`**](nodule_analysis_pipline.py): Implements the entire pipeline, from segmentation to nodule grouping, analysis, and diagnosis.

### 4. Nodule Classification 

<p align="center">
  <img src="READMD-Assets\Classifier_architecture.png" alt="CNN cls architecture" title="CNN cls architecture" width="600" />
</p>

After grouping the potential nodules detected by the segmentation model, the next step is to screen out any false positives using a CNN-based binary classification network. This step aims to refine the detection by distinguishing between actual nodules and non-nodules. All components for this subsystem are implemented from scratch, including:

- A custom dataset with data balancing, GPU-based augmentation, and disk pre-caching to accelerate training.
- A fully-featured command-line application for training the classifier, with functionality for fine-tuning the malignancy classifier in the following step.

Relevant files:

- [**`utils/NoduleClassifier.py`**](utils/NoduleClassifier.py): Implements the CNN architecture used for nodule classification.
- [**`utils/classifier_dset.py`**](utils/classifier_dset.py): Defines the custom dataset class for nodule classification, including data loading and balancing.
- [**`utils/classifier_dset_utils.py`**](utils/classifier_dset_utils.py): Provides utilities for dataset augmentation.
- [**`utils/nodule_precaching.py`**](utils/nodule_precaching.py): Implements disk pre-caching functionalities to speed up the training process.
- [**`classifier_training.py`**](classifier_training.py): Implements the command-line application for training the classification model.



### 5. Diagnosis and Malignancy Classification

In the final step, the classification results are aggregated to make a comprehensive diagnosis. The system evaluates each identified nodule using a dedicated CNN classifier to determine whether it is malignant or benign. This malignancy classifier is fine-tuned from the nodule classifier to enhance accuracy in distinguishing between benign and malignant nodules. The entire pipeline culminates in generating a report that flags patients as suspicious if at least one nodule is identified as potentially malignant.

Relevant files:

- [**`utils/classifier_dset.py`**](utils/classifier_dset.py): Defines the custom dataset class for malignancy classification, including data loading and balancing.
- [**`classifier_training.py`**](classifier_training.py): Implements the command-line application for training the malignancy classification model.
- [**`nodule_analysis_pipline.py`**](nodule_analysis_pipline.py): Implements the entire pipeline, from segmentation to nodule grouping, analysis, and diagnosis.



## Dataset: LUNA 2016

This project uses the **LUNA 2016 (LUng Nodule Analysis)** dataset, which consists of 3D CT scans labeled with lung nodule annotations. The dataset is part of a challenge aimed at improving nodule detection algorithms through standardized evaluation. It includes 10 subsets of scans for tasks like **Nodule Detection (NDET)** and **False Positive Reduction (FPRED)**. 

For more details, visit the [LUNA 2016 Grand Challenge](https://luna16.grand-challenge.org/Description).

## Installation

To set up the Lung Nodule Analysis System, follow these steps:

1. **Clone the Repository**
   
   ```bash
   git clone git@github.com:joyou159/Lung-Nodule-Analysis-System.git
   cd Lung-Nodule-Analysis-System 
    ```
2. **Install Dependencies** 

   ```bash
    pip install -r requirements.txt
    ```
3. **Run Infernece**

    ```bash
    # check --help to get all possible command-line arguments 
    python nodule_analysis_app.py "subject's series_uid" 
    ```

## Models Training


For training, I utilized the Kaggle platform to develop
 both the classification and segmentation models.
fortunately, half of the dataset, **specifically subsets 0 through 4**, is hosted on Kaggle, which allowed me to set up a notebook and began working directly.


<table align="center">
    <tr>
        <td align="center">
            <a href="https://www.kaggle.com/code/joyou159/nodule-malignancy-classifiers">
                <img src="READMD-Assets/Kaggle_logo.png" alt="Classification Models" width="100" />
            </a>
            <p>Classification Models</p>
        </td>
        <td align="center">
            <a href="https://www.kaggle.com/code/joyou159/nodule-detection">
                <img src="READMD-Assets/Kaggle_logo.png" alt="Segmentation Model" width="100" />
            </a>
            <p>Segmentation Model</p>
        </td>
        <td align="center">
            <a href="https://www.kaggle.com/code/joyou159/luna-pipline/notebook">
                <img src="READMD-Assets/Kaggle_logo.png" alt="Analysis Pipeline" width="100" />
            </a>
            <p>Analysis Pipeline</p>
        </td>
    </tr>
</table>

Each notebook works as a standalone module, so feel free to investigate each one individually.

During the training process, i employed TensorBoard to monitor model metrics and performance while identifying any potential issues. You can explore these insights by launching the
 [`tensorboard.ipynb`](tensorboard.ipynb) notebook after specifying the runs file.


> [!NOTE]  
> Due to limited on-disk space on Kaggle (≈ 19.5GB) and computational resources quota, the pipeline models were trained using only one subset of the LUNA 2016 dataset instead of the entire dataset. This limitation was necessary for on-disk dataset pre-caching to ensure high-speed training. As a result, the model's performance may be somewhat lower than if it had been trained on the complete dataset. If you have sufficient computational resources, we encourage you to train the models on the entire dataset, as this would significantly enhance performance.

### Models Retraining Setup 

Before retraining these models, ensure that you adjust the file paths in the source code:

- Update the path to [`malignancy_annotations\annotations_for_malignancy.csv`](malignancy_annotations\annotations_for_malignancy.csv) in the source code to match your local setup.

- Set the dataset directory paths as needed to reflect your environment. 


1. **Classification Models**
    
    ```bash
    # precaching to speed up training process
    python utils\nodule_precaching.py \
      --num-workers 4 \
      --batch-size 64 \
      --subsets-included 0 \  # include more subsets if possible
    ```

    ```bash
    # Use --help to get more details about each argument
    # training nodule classifier
    python classifier_training.py \
      --num-workers 4 \
      --batch-size 128 \
      --epochs 10 \
      --subsets-included 0 \  # include more subsets if possible
      --augmented \
      --augment-flip \
      --augment-offset \
      --augment-scale \
      --augment-rotate \
      --augment-noise
    ```

    ```bash
    # Use --help to get more details about each argument
    # malignancy classifier fine-tuning
    python classifier_training.py \
      --num-workers 4 \
      --batch-size 64 \
      --malignant \
      --dataset MalignantLunaDataset \
      --finetune nodule_cls_model_path \  
      --finetune-depth 1 \
      --epochs 10 \
      --subsets-included 0 \  # include more subsets if possible
    ```

2. **Segmentation Model** 
    
    ```bash
    # precaching to speed up training process
    python utils\seg_precaching.py \
      --num-workers 4 \
      --batch-size 64 \
      --subsets-included 0 \  # include more subsets if possible
    ```

    ```bash
    # try --help to get more details about each argument
    python seg_training.py \
      --num-workers 4 \
      --batch-size 128 \
      --epochs 10 \
      --subsets-included 0 \  # include more subsets if possible
      --augmented \
      --augment-flip \
      --augment-offset \
      --augment-scale \
      --augment-rotate \
      --augment-noise
    ```

## Acknowledgments

Inspired by **"Deep Learning with PyTorch"** (Part 2) by Eli Stevens, Luca Antiga, and Thomas Viehmann, Manning Publications.

