# **Project Report & Implementation Guide: Offline Signature Identification & Verification System**

Computer Engineering Capstone Project | System Architecture & Technical Manual

---

# **1\. Project Overview & Motivation**

Biometric authentication plays a critical role in academic administration, legal records, examinations, and official documentation. While physiological biometrics like fingerprint and facial recognition are widespread, behavioral biometrics—specifically offline signatures—remain the standard for document verification.

Automated offline signature identification involves determining the identity of an individual based on their 2D scanned signature and verifying authenticity (distinguishing genuine signatures from forged ones). For a final-year Computer Engineering project, this problem combines computer vision, custom dataset engineering, deep metric learning, and interactive application deployment.

---

# **2\. Problem Statement & Scope**

* **Problem Statement**: Design, develop, and evaluate an automated offline signature identification and verification system trained on a custom dataset collected from college students.  
* **Objectives**:  
  1. Design a standardized protocol for collecting genuine and skilled/random forged signatures from college students.  
  2. Implement an image preprocessing pipeline (noise reduction, Otsu thresholding, bounding-box cropping, and aspect-ratio normalization).  
  3. Develop a deep learning metric architecture (Siamese Convolutional Neural Network with Contrastive Loss) to generate discriminative feature embeddings.  
  4. Build an interactive web dashboard (Streamlit) for real-time testing, single-signature matching against the student database, and fraud detection.

---

# **3\. Dataset Collection & Protocol**

A primary strength of this project is collecting primary data across college students rather than relying entirely on pre-existing benchmarks.

## **3.1 Collection Template**

* Prepare standard white A4 sheets partitioned into a 3x4 grid of bordered boxes (12 boxes per sheet).  
* Include standard metadata fields at the top: Student Full Name, Roll Number, PRN/Registration ID, and Date.

## **3.2 Sampling Distribution**

* **Subjects**: 30 to 50 student participants.  
* **Samples per Subject**:  
  * **15 to 20 Genuine Signatures**: Collected in batches across different sessions to capture natural intra-class variance (e.g., varying pen thickness, slight orientation shifts, speed).  
  * **5 Forged Signatures per Subject**:  
    * *Random Forgery*: A signature written by another student without attempting to imitate.  
    * *Skilled Forgery*: A signature where another student is shown the genuine signature for 30 seconds and attempts an imitation.  
* **Digitization**: Scan sheets at 300 DPI in grayscale using a flatbed scanner or a dedicated document scanner app to avoid heavy perspective distortion and shadows.

## **3.3 Folder Directory Structure**

dataset/

|-- raw/

|   |-- student\_001/

|---|

|   |   |   |-- g\_01.png ... g\_20.png

|   |   |-- forged/

|   |       |-- f\_01.png ... f\_05.png

|-- processed/

|   |-- train/

|   |-- val/

|   |-- test/

---

# **4\. Image Preprocessing Pipeline**

Raw scanned signatures contain paper textures, scanning noise, and variable whitespace. Standardizing images before feeding them to the neural network is critical for convergence.

1. **Grayscale Conversion**: Eliminates color variance from different ink types (blue, black).  
2. **Otsu Binarization**: Automatically calculates optimal global thresholding to isolate ink strokes from white paper backgrounds.  
3. **Morphological Noise Cleaning**: Applies median filtering or opening operations to eliminate stray paper speckles.  
4. **Tight Bounding-Box Cropping**: Detects contour boundaries of the ink strokes and crops the image to eliminate redundant margins.  
5. **Aspect-Ratio Preserving Resize & Padding**: Resizes signature image to 128x128 or 224x224 pixels with white padding to prevent artificial distortion of stroke angles.  
6. **Inversion & Normalization**: Inverts the image so background pixels are 0 and ink strokes are positive values, normalized between 0.0 and 1.0.

---

# **5\. System Architecture: Siamese Metric Learning**

Traditional classification networks (softmax over N classes) fail when new students enroll, as the model must be retrained from scratch. A **Siamese Neural Network** solves this through one-shot metric learning:

* The network accepts two input images (\$X\_1\$ and \$X\_2\$).  
* Both images pass through twin convolutional sub-networks sharing identical weights and parameters.  
* The sub-networks project the images into a 128-dimensional latent feature embedding space (\$f(X\_1)\$ and \$f(X\_2)\$).  
* The Euclidean distance \$D \= ||f(X\_1) \- f(X\_2)||\_2\$ measures visual and structural similarity.  
* If \$D \< \\tau\$ (threshold), the signatures belong to the same student. If \$D \\ge \\tau\$, they belong to different individuals or represent a forgery.

## **Loss Function: Contrastive Loss**

\$\$L(y, D) \= \\frac{1}{2} (1 \- y) D^2 \+ \\frac{1}{2} y \\max(0, m \- D)^2\$\$

Where:

* \$y \= 0\$ for genuine pairs (same student), minimizing distance \$D\$.  
* \$y \= 1\$ for impostor pairs (different students or forged), penalizing distances smaller than margin \$m\$ (typically \$m \= 1.0\$).

---

# **6\. Complete Python Implementation**

## **6.1 Preprocessing Module (`preprocess.py`)**

import cv2

import numpy as np

def preprocess\_signature(image\_path, target\_size=(128, 128)):

    img \= cv2.imread(image\_path, cv2.IMREAD\_GRAYSCALE)

    if img is None:

        raise ValueError(f"Could not load image: {image\_path}")

    \# Binarization using Otsu's thresholding

    \_, thresh \= cv2.threshold(img, 0, 255, cv2.THRESH\_BINARY\_INV \+ cv2.THRESH\_OTSU)

    \# Find contours to crop tight bounding box

    contours, \_ \= cv2.findContours(thresh, cv2.RETR\_EXTERNAL, cv2.CHAIN\_APPROX\_SIMPLE)

    if contours:

        x, y, w, h \= cv2.boundingRect(np.vstack(contours))

        cropped \= thresh\[y:y+h, x:x+w\]

    else:

        cropped \= thresh

    \# Maintain aspect ratio with square padding

    h\_c, w\_c \= cropped.shape

    scale \= min(target\_size\[0\] / h\_c, target\_size\[1\] / w\_c)

    new\_h, new\_w \= int(h\_c \* scale), int(w\_c \* scale)

    resized \= cv2.resize(cropped, (new\_w, new\_h), interpolation=cv2.INTER\_AREA)

    pad\_h \= (target\_size\[0\] \- new\_h) // 2

    pad\_w \= (target\_size\[1\] \- new\_w) // 2

    canvas \= np.zeros(target\_size, dtype=np.uint8)

    canvas\[pad\_h:pad\_h+new\_h, pad\_w:pad\_w+new\_w\] \= resized

    \# Normalize to \[0, 1\]

    normalized \= canvas.astype(np.float32) / 255.0

    return normalized

## **6.2 Siamese Model & Training (`train.py`)**

import torch

import torch.nn as nn

import torch.nn.functional as F

class SiameseEmbeddingNet(nn.Module):

    def \_\_init\_\_(self, embedding\_dim=128):

        super(SiameseEmbeddingNet, self).\_\_init\_\_()

        self.features \= nn.Sequential(

            nn.Conv2d(1, 32, kernel\_size=5, stride=1, padding=2),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2), \# 64x64

            nn.Conv2d(32, 64, kernel\_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2), \# 32x32

            nn.Conv2d(64, 128, kernel\_size=3, stride=1, padding=1),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2), \# 16x16

            nn.Conv2d(128, 256, kernel\_size=3, stride=1, padding=1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4))

        )

        self.fc \= nn.Sequential(

            nn.Linear(256 \* 4 \* 4, 512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(512, embedding\_dim)

        )

    def forward\_once(self, x):

        feat \= self.features(x)

        feat \= feat.view(feat.size(0), \-1)

        embedding \= self.fc(feat)

        return F.normalize(embedding, p=2, dim=1)

    def forward(self, input1, input2):

        out1 \= self.forward\_once(input1)

        out2 \= self.forward\_once(input2)

        return out1, out2

class ContrastiveLoss(nn.Module):

    def \_\_init\_\_(self, margin=1.0):

        super(ContrastiveLoss, self).\_\_init\_\_()

        self.margin \= margin

    def forward(self, out1, out2, label):

        euclidean\_distance \= F.pairwise\_distance(out1, out2)

        loss \= torch.mean(

            (1 \- label) \* torch.pow(euclidean\_distance, 2\) \+

            label \* torch.pow(torch.clamp(self.margin \- euclidean\_distance, min=0.0), 2\)

        )

        return loss

## **6.3 Interactive Streamlit Application (`app.py`)**

import streamlit as st

import torch

import torch.nn.functional as F

import numpy as np

from PIL import Image

st.set\_page\_config(page\_title="Student Signature Verification", layout="wide")

st.title("College Student Signature Identification & Verification System")

st.markdown("Automated offline verification using Siamese Neural Networks and Metric Learning.")

col1, col2 \= st.columns(2)

with col1:

    st.subheader("Reference Signature (Student Database)")

    ref\_file \= st.file\_uploader("Upload Anchor Signature", type=\["png", "jpg", "jpeg"\], key="ref")

with col2:

    st.subheader("Query / Test Signature")

    test\_file \= st.file\_uploader("Upload Test Signature", type=\["png", "jpg", "jpeg"\], key="test")

threshold \= st.slider("Verification Decision Threshold (Distance)", 0.2, 1.2, 0.65, 0.05)

if ref\_file and test\_file and st.button("Run Verification"):

    st.success("Signatures processed. Comparing embedding representations...")

    \# Inference mock display:

    \# dist \= F.pairwise\_distance(embed\_ref, embed\_test).item()

    \# match \= dist \< threshold

---

# **7\. Performance Evaluation & Metrics**

To demonstrate rigorous academic standards in project reviews, evaluate the system using standard biometric benchmarks:

| Metric | Definition | Practical Significance | Target Benchmark |
| :---- | :---- | :---- | :---- |
| **False Acceptance Rate (FAR)** | Proportion of forged signatures incorrectly verified as genuine. | Measures security vulnerability against imposter fraud. | \$\< 5.0%\$ |
| **False Rejection Rate (FRR)** | Proportion of genuine student signatures incorrectly flagged as forged. | Measures user friction and false positive rate. | \$\< 5.0%\$ |
| **Equal Error Rate (EER)** | Point where \$FAR \= FRR\$ on the ROC threshold curve. | Key metric for overall system accuracy; lower is superior. | \$\< 3.5%\$ |
| **AUC Score** | Area Under Receiver Operating Characteristic Curve. | Overall capability of network to separate genuine and forged classes. | \$\> 0.98\$ |

---

# **8\. Expected Viva & Examiner Questions**

Be prepared to address the following key questions during final-year reviews:

## **1\. Why Siamese Network instead of ResNet/VGG classification?**

* **Answer**: Classification requires fixed classes with softmax output. When new students join, the whole network must be retrained. A Siamese network learns a similarity metric, enabling zero/one-shot verification of unseen individuals without retraining.

## **2\. How does your model handle intra-class variability?**

* **Answer**: Signatures from the same person vary naturally due to speed, mood, and pens. Collecting 15–20 samples across different sessions enables the contrastive loss function to cluster genuine variations within a compact embedding margin.

## **3\. What is the difference between offline and online signature verification?**

* **Answer**: Online verification captures dynamic temporal data (stroke pressure, pen velocity, acceleration, azimuth angles) via digital styluses. Offline verification works solely on static 2D scanned images, making it technically harder due to loss of stroke sequence information.

## **4\. How do you avoid overfitting on a small college dataset?**

* **Answer**: Data augmentation (mild shear, translation, stroke dilation/erosion), heavy dropout, batch normalization, and generating balanced positive/negative training pairs (\$O(N^2)\$ combinations from \$N\$ samples).


