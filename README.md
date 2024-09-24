# AI-Dental-Imaging-Detection

This repository contains the machine learning models developed for dental image classification and detection. The goal of the project is to train AI models to detect and classify various dental conditions, including oral lesions, cavities, fillings, and plaque, using advanced image processing techniques.

## Project Overview

The project consists of two main components:
1. **Oral Lesions Project**: Using CNN for image classification to detect eight different types of oral lesions.
2. **Oral Condition Project**: Implementing YOLOv5 for object detection of dental conditions (plaque, fillings, and cavities) and using UNet for semantic segmentation to assess the severity of the conditions.

### Key Features:
- **Image Classification**: Common CNN model for detecting oral lesions with over 94% accuracy.
- **Object Detection**: Fine-tuning a YOLOv5 model to detect dental conditions in images.
- **Semantic Segmentation**: Using UNet to segment oral conditions for further analysis.

## Models Used

1. **CNN (Convolutional Neural Network)**: Used for classifying oral lesions.
2. **YOLOv5 (You Only Look Once)**: Applied for object detection to identify various oral conditions.
3. **UNet**: Used for semantic segmentation to evaluate the severity of detected conditions.

## Project Structure

- `src/`: Contains the source code for model training and evaluation.
- `data/`: Placeholder for the dental image datasets (not included due to size constraints).
- `notebooks/`: Jupyter notebooks for training models and conducting experiments.
- `models/`: Pre-trained models and weights (if available).
- `results/`: Contains model outputs, performance metrics, and visualizations.

## Dataset

Two main datasets are used in this project:
1. **Oral Lesions Dataset**: Contains annotated images of oral lesions.
2. **Oral Conditions Dataset**: Includes images with annotations for plaque, cavities, and fillings.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SLAM-CROC/AI-innovate-Dental-Imaging.git
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Download the datasets and place them in the `data/` folder.
4. Train the models using the provided notebooks:
 - `oral_lesions_classification.ipynb`
 - `oral_condition_detection_yolov5.ipynb`
 - `semantic_segmentation_unet.ipynb`
5. View results and predictions in the `results/` folder.

## Results
- **Oral Lesion Classification**: Achieved ~94% accuracy on the testing set.
- **Object Detection (YOLOv5)**: Successfully detected dental conditions with good precision and recall.
- **Semantic Segmentation (UNet)**: Segmented images with varying levels of success, limited by the dataset size and manual labeling.
