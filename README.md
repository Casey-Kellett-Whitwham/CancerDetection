# 🧠 Using Deep Learning to Detect Cancer Types and Variants

This project was completed as part of my Senior Project at **Luther College**.  
The goal was to develop a robust deep learning pipeline capable of detecting various cancer types and their variants using medical imaging data.

---

## 📊 Exploratory Data Analysis (EDA)

- **Class Distribution**  
  ![image](https://github.com/user-attachments/assets/91707e36-12f6-42eb-8054-ac47842f7c6b)
  
- **Sample Images by Scan Type**  
  ![image](https://github.com/user-attachments/assets/99dcf429-16ef-4dd0-8b98-dab9218e0016)

---

## 📁 Project Structure

The project was divided into **six distinct phases**, each building on the previous:

### 🔹 Phase 1: Initial Model Testing
- Implemented a custom CNN using TensorFlow.
- Tested baseline performance on raw data.
- **Results**:
  - Accuracy: `81.79%`
- **Key Insight**: Strong overfitting (Train Accuracy - 97.02%) indicated a need for regularization or data augmentation.

### 🔹 Phase 2: Data Augmentation
- Applied transformations (rotation, flipping, zooming) to improve generalization.
- **Results**:
  - Improved validation accuracy to `84.27%`.

### 🔹 Phase 3: Hyperparameter Optimization with HyperBand
- Used KerasTuner's HyperBand to optimize CNN structure and training parameters.
- **Results**:
  - Best model accuracy: `92%`

### 🔹 Phase 4: Switched to ResNet50
- Replaced the custom CNN with pretrained ResNet50 (with fine-tuning).
- **Results**:
  - Accuracy: `98.37%`
  - Notable performance boost and faster convergence.

### 🔹 Phase 5: Grouped by Scan Type + 4 CNN Models
- Dataset was split by scan type (e.g., MRI/CT, histopathology,Pap Smear & Blood Smear).
- Trained 4 separate custom CNNs for each type.
- **Results**:
  - Per-scan-type accuracy (CNN):
    - MRI/CT: `96.02%`
    - Histopathology: `85.01%`
    - PAP: `99.42%`
    - Blood: `99.96%`

### 🔹 Phase 6: ResNet50 Models by Scan Type
- Same scan-based split, but models replaced with ResNet50 for each group.
- **Results**:
  - Per-scan-type accuracy (ResNet50):
    - MRI/CT: `99.4%`
    - Histopathology: `98.33%`
    - PAP: `100%`
    - Blood: `99.38%`

- **Comparison Table**:

| Scan Type      | CNN Accuracy | ResNet50 Accuracy |
|----------------|--------------|-------------------|
| MRI/CT            | `96.02%`         | `99.4%`              |
| Histopathology             | `85.01%`         | `98.3%`              |
| Pap Smear | `99.42%`         | `100%`              |
| Blood Smear            | `99.96%`         | `99.38%`              |

---

## 🎥 DEMO

<p align="center">
  <div style="display: inline-block; text-align: center; width: 48%; margin: 1%;">
    <img src="https://github.com/user-attachments/assets/bb00d56b-a760-430b-9d80-de5bf09abc96" alt="Video Demo of Prediction" width="100%" />
    <br />
    <em>Prediction Interface Demo</em>
  </div>

  <div style="display: inline-block; text-align: center; width: 48%; margin: 1%;">
    <img src="https://github.com/user-attachments/assets/be48db1b-0dd0-47cd-9687-c69f200fb624" alt="Model Attention Heatmap" width="100%" />
    <br />
    <em>Model Attention (Grad-CAM Heatmap)</em>
  </div>
</p>

---

## ✅ Key Takeaways

- Pretrained architectures like ResNet50 significantly outperformed custom CNNs.
- Grouping data by scan type led to more accurate and specialized models.
- Grad-CAM visualizations improved model interpretability and trustworthiness.

---

## 🧰 Tools & Libraries Used

- Python
- TensorFlow & Keras
- KerasTuner
- Matplotlib & Seaborn
- Pytorch

---

## 📎 Acknowledgements

- Luther College Faculty Advisors
- Publicly available MultiCancer Dataset from [Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/obulisainaren/multi-cancer))
