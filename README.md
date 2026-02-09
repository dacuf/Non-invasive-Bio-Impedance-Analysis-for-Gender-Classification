# Non-invasive-Bio-Impedance-Analysis-for-Gender-Classification
Non-invasive poultry gender classification using Bio-Impedance Spectroscopy and SVM. Achieved 80.95% accuracy on Day 11 of incubation.
# Non-invasive Chicken Egg Gender Classification using Bio-Impedance & SVM

## 📌 Project Overview
In the industrial poultry sector, identifying the gender of chicken eggs before hatching is crucial for economic efficiency and ethical reasons. This project proposes a **non-invasive method** using **Bio-Impedance Spectroscopy** combined with **Machine Learning** to classify eggs as male or female.

Unlike traditional invasive methods, this approach analyzes the conductivity variations of the embryo inside the shell, preserving the egg's integrity.

## Experimental Setup & Data Acquisition
The dataset was collected using a precision **LCR Meter** setup on Taiwan Native Chicken eggs.

* **Dataset:** 35 eggs (21 Male, 13 Female, 1 Non-viable).
* **Method:** 4-Electrode configuration.
    * Electrodes were attached to the shell in a specific symmetrical layout.
    * Impedance was measured across pairs: **1-2, 2-3, 2-4, 3-4**.
* **Duration:** Measurements were taken daily from **Day 1 to Day 14** of incubation.

> **Note:** The raw data collection was performed by CN. Nguyen Ngoc Thao Nhi (Taiwan). My role focused on **Signal Processing, Feature Engineering, and Model Development**.

<img width="725" height="268" alt="image" src="https://github.com/user-attachments/assets/28aa0608-21b6-4a0d-830e-6569582ea6dd" />


##  Methodology

### 1. Signal Processing & Physics
* **Feature Selection:** Analyzed impedance trends across all channel pairs.
* **Physical Insight:** The pairs **Z23/Z24** (transversal measurements) were identified as the most critical features. These paths cross the embryo directly, capturing the physiological conductivity differences between male and female development.

### 2. Machine Learning Pipeline
* **Algorithm:** Support Vector Machine (SVM).
* **Kernel:** Radial Basis Function (RBF) for non-linear separation.
* **Validation:** * **K-Fold Cross Validation (n=7)** was used to ensure reliability given the small dataset size.
    * **Test Size:** 40% (0.4).

## 📊 Key Results

The model achieved peak performance on **Day 11 of incubation**, which was identified as the "Critical Window" for gender differentiation.
<img width="741" height="556" alt="image" src="https://github.com/user-attachments/assets/e7a91264-78d9-4e8a-982e-09b54664ff04" />
| Metric | Result |
| :--- | :--- |
| **Accuracy** | **80.95%** |
| **Best Day** | Day 11 |
| **Key Features** | Impedance Z23/Z24 |

### Handling Imbalanced Data
The dataset was naturally imbalanced (21 Males vs. 13 Females). The SVM hyperparameters (C, Gamma) were fine-tuned to penalize misclassification of the minority class, ensuring the model didn't just bias towards the majority.

## Tech Stack
* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib/Seaborn.
* **Hardware Concepts:** LCR Meter, Bio-Impedance Analysis, 4-Terminal Sensing.

---
*This project was part of my University Thesis focusing on Applied Engineering Physics.*
