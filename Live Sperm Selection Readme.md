# Live Sperm Selection Using Computer Vision and Deep Learning

## Overview

Live Sperm Selection is an intelligent computer vision-based framework developed to automatically detect, track, analyze, and select viable sperm cells from microscopic video sequences. The system utilizes image processing, object tracking, and machine learning techniques to assist fertility specialists in identifying high-quality sperm for Assisted Reproductive Technologies (ART) such as IVF and ICSI.

The objective of this research is to improve sperm selection accuracy, reduce manual effort, and provide objective fertility assessment through automated analysis.

---

## Features

- Real-time sperm detection
- Live sperm tracking
- Motility analysis
- Velocity estimation
- Sperm trajectory visualization
- Quality assessment and ranking
- Automated live sperm selection
- Research-oriented performance evaluation
- Video-based microscopic analysis

---

## Installation

### Clone Repository

```bash
git clone https://github.com/BATINIDHANWANTH/Live-Sperm-Seletion-Thesis.git
cd Live-Sperm-Seletion-Thesis
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run Detection Module
## Running Procedure

### Step 1: Dataset Preparation

Organize the sperm microscopy images and videos into the required dataset structure.

```bash
Dataset/
├── train/
├── valid/
└── test/
```

---

### Step 2: YOLO Model Training

The project contains multiple Jupyter notebooks for training different YOLO model variants (YOLOv8 – YOLOv24) with data augmentation techniques.

Open and execute the notebooks sequentially:

```bash
YOLOv8_Training.ipynb
YOLOv9_Training.ipynb
YOLOv10_Training.ipynb
...
YOLOv24_Training.ipynb
```

The training process includes:

- Dataset preprocessing
- Data augmentation
- Model training
- Hyperparameter tuning
- Performance evaluation
- Best weight generation

The trained weights are stored in:

```bash
runs/detect/
```

---

### Step 3: Sperm Detection and Motion Analysis

Run the main application:

```bash
python app.py
```

#### Functions Performed

- Microscopic video loading
- Sperm head detection using trained YOLO models
- ROI (Region of Interest) extraction
- Multi-object sperm tracking
- Sperm motility analysis
- Velocity estimation
- Movement trajectory analysis
- Progressive motility assessment
- Visualization of sperm movement patterns

#### Outputs

- Detected sperm heads
- ROI images
- Tracking trajectories
- Speed statistics
- Motility measurements

---

### Step 4: Sperm Morphology Analysis

Execute:

```bash
python "SDF Classification.py"
```

#### Functions Performed

- Sperm morphology feature extraction
- Head shape analysis
- Morphological quality assessment
- Sperm DNA Structure Evaluation

#### Morphology Parameters

- Head Length
- Head Width
- Head Area
- Head Perimeter
- Shape Regularity
- Morphological Score

---

### Step 5: DNA Fragmentation Prediction

The morphology and motion features are further utilized to predict:

### DNA Fragmentation Index (DFI)

The model estimates:

- DNA Fragmentation
- No DNA Fragmentation


This assists fertility specialists in identifying sperm with better fertilization potential.

---

## Complete Pipeline

```text
Microscopic Video Input
          │
          ▼
      YOLO Detection
      (YOLOv8–YOLOv24)
          │
          ▼
    Sperm Head Detection
          │
          ▼
      ROI Extraction
          │
          ▼
     Object Tracking
          │
          ▼
   Motility & Speed Analysis
          │
          ▼
  Movement Pattern Analysis
          │
          ▼
 Morphology Feature Extraction
          │
          ▼
     SDF Classification
          │
          ▼
 DNA Fragmentation Prediction
          │
          ▼
     Fertility Assessment
```

---

## Generated Outputs

The framework generates:

- Sperm Head Detection Results
- ROI Images
- Motility Reports
- Velocity Analysis
- Movement Trajectories
- Morphology Measurements
- SDF Classification Results
- DNA Fragmentation Index (DFI) Prediction
- Fertility Assessment Reports
## Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| OpenCV | Image Processing |
| NumPy | Numerical Computation |
| Pandas | Data Analysis |
| Matplotlib | Visualization |
| Scikit-Learn | Machine Learning |
| TensorFlow/PyTorch | Deep Learning |

---

## Applications

- In Vitro Fertilization (IVF)
- Intracytoplasmic Sperm Injection (ICSI)
- Fertility Clinics
- Biomedical Research
- Computer-Assisted Semen Analysis (CASA)

---

## Future Enhancements

- Real-time microscope integration
- Deep learning-based segmentation
- Cloud-based fertility assessment
- Explainable AI for sperm selection
- Mobile deployment support

---

## Research Contribution

This project contributes to automated sperm quality assessment by integrating computer vision and machine learning techniques into a unified framework for intelligent live sperm selection.

---

## Author

**B. Dhanwanth**

GitHub: https://github.com/BATINIDHANWANTH

---

## License

This project is developed for academic and research purposes.

Feel free to use and extend this work with proper citation.