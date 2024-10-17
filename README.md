# Lung Nodule Analysis System 


<p align="center">
  <img src="READMD-Assets\Project_Steps.png" alt="The project outlines" title="The project outlines" width="700" />
</p>

This project is an end-to-end deep learning pipeline for lung cancer detection using **3D CT** scan data. Through five-stages pipline we seek to detect candidate lumps in the lung that potentially look like nodules using segmentation, then screening them out using cascade of tailored CNN classifiers for malignancy

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
  <img src="READMD-Assets\Classifier_architecture.png" alt="CNN cls architecture" title="CNN cls architecture" width="500" />
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
   
   First, clone the repository to your local machine:
   ```bash
   git clone git@github.com:joyou159/Lung-Nodule-Analysis-System.git
   cd Lung-Nodule-Analysis-System 
    ```
2. **Install Dependencies** 

   First, clone the repository to your local machine:
   ```bash
    pip install -r requirements.txt
    ```

## Models Training

> [!NOTE]  
> Due to limited computational resources, the pipeline models were trained using only one subset of the LUNA 2016 dataset rather than the full dataset. Consequently, the model's performance may be somewhat lower than if it were trained on the complete dataset. If you have sufficient computational resources, you are encouraged to train the models on the entire dataset, as this would significantly enhance performance. 









