```markdown
# Meteorological and Satellite-Based Precipitation Prediction

## Objective
This project classifies precipitation into four categories (**No Rain**, **Light Rain**, **Medium Rain**, and **High Rain**) by fusing:
1. **Meteorological (tabular time-series) data**  
2. **Satellite imagery** from GOES data  

A variety of model architectures are explored, including a baseline RNN and hybrid deep-learning models (e.g., ConvLSTM2D + LSTM).

---

## Folder Structure
```
Meteorologic...Lake-Michigan/
├── demo_app.py
├── images/
│   └── ... (any relevant images, figures, or sample outputs)
├── LICENSE
├── model_architecture.png
├── Notebook_0_Data_Preparation_for_Inference.ipynb
├── Notebook_1_Inference.ipynb
├── Notebook_2_EDA.ipynb
├── Notebook_3_Data_Preprocessing.ipynb
├── Notebook_4_Baseline_RNN_Model.ipynb
├── Notebook_5_24_Window_Hybrid_Model_Automatic_Weights.ipynb
├── Notebook_6_24_Window_Hybrid_Model_Manual_Weights.ipynb
├── Notebook_7_48_Window_Hybrid_Model_Automatic_Weights.ipynb
├── Notebook_8_48_Window_Hybrid_Model_Manual_Weights.ipynb
├── Notebook_9_72_Window_Hybrid_Model_Automatic_Weights_Additional_10_Epochs.ipynb
├── Notebook_10_72_Window_Hybrid_Model_Automatic_Weights.ipynb
├── Notebook_11_72_Window_Hybrid_Model_Manual_Weights_Additional_10.ipynb
├── Notebook_12_72_Window_Hybrid_Model_Manual_Weights.ipynb
└── README.md
```

---

## Notebooks Overview

1. **Notebook_0_Data_Preparation_for_Inference.ipynb**  
   - Prepares the data (both meteorological features and satellite images) for **inference** in the trained models.  
   - Shapes and formats the test/validation data for batch prediction.

2. **Notebook_1_Inference.ipynb**  
   - Loads each trained model (baseline RNN and hybrid variants) and runs inference.  
   - Calculates evaluation metrics (accuracy, F1-score) and confusion matrices for final comparison.  
   - Summarizes all results.

3. **Notebook_2_EDA.ipynb** (Exploratory Data Analysis)  
   - Explores **GOES Satellite** data characteristics (pixel intensity distributions, temporal coverage).  
   - Visualizes sample images and checks for patterns or anomalies.

4. **Notebook_3_Data_Preprocessing.ipynb**  
   - Performs data cleaning (handles missing values, drops irrelevant columns).  
   - **Resizes** satellite images to the required resolution (64×64).  
   - **Creates the precipitation_category** target variable based on thresholds.  
   - Implements **sliding window** logic to form sequences of meteorological data and corresponding satellite frames (e.g., 24-hour, 48-hour, and 72-hour inputs).

5. **Notebook_4_Baseline_RNN_Model.ipynb**  
   - Trains a simple RNN model (LSTM-based) using only **meteorological** time-series data.  
   - Establishes a baseline performance for precipitation-category classification.

6. **Notebooks 5 to 12: Hybrid Models**  
   - Each notebook trains a **hybrid** model that combines:
     - **Meteorological features** (via LSTM layers, sometimes with Attention)  
     - **Satellite image sequences** (via ConvLSTM2D or other 3D/Conv approaches)  
   - Different window sizes (24, 48, 72 hours) and weight-integration methods (automatic vs. manual) are tested.  
   - **Example architectures** tested:
     - *ConvLSTM2D + LSTM (Shallow/Deep)*  
     - *Conv3D + ConvLSTM2D + LSTM (Shallow/Deep)*  
   - Notebooks 5 & 6 focus on **24-hour** data, 7 & 8 on **48-hour**, while 9–12 explore **72-hour** sequences with additional epochs and/or manual weighting strategies.

---

## Hybrid Model Architecture Explanation

Below is an overview of the **hybrid** architecture (as shown in `model_architecture.png`):

1. **Meteorological Branch**  
   - **Input** (`meteo_input`): shape `(None, T, d)`, where `T` is the time-window (e.g., 72 hours) and `d` is the number of meteorological features (e.g., 10).  
   - **Stacked LSTM layers** (e.g., `lstm_17`, `lstm_18`): Extract temporal patterns from the input sequences.  
   - **Dropout layers** follow the LSTMs to reduce overfitting (e.g., `dropout_26`, `dropout_27`).  
   - **Attention layer** (e.g., `attention_4`): Learns to focus on the most relevant time steps within the meteorological sequence.  
   - **Final LSTM** (e.g., `lstm_19`): Aggregates the attention‐weighted features into a single vector.  
   - **Batch Normalization** to stabilize training (e.g., `batch_normalization_9`).

2. **Satellite/Cloud Branch**  
   - **Input** (`cloud_input`): shape `(None, T_img, 64, 64, 1)`, where `T_img` is the number of satellite frames (e.g., 24 frames), and `64×64` is the resized spatial dimension with 1 channel (grayscale).  
   - **ConvLSTM2D layers** (e.g., `conv_lstm2d_8` and `conv_lstm2d_9`): Extract **spatiotemporal** features from the sequence of images.  
   - **Dropout layers** (`dropout_24`, `dropout_25`) to prevent overfitting.  
   - **TimeDistributed** layer (`time_distributed_4`): Applies further transformations across each time slice of the feature maps.  
   - **LSTM** (e.g., `lstm_16`): Interprets the final, flattened feature sequence.  
   - **Batch Normalization** (e.g., `batch_normalization_8`) to stabilize training.

3. **Concatenation and Final Dense Layers**  
   - The **Meteorological** and **Satellite** branches each produce a feature vector (`(None, 64)` in this design).  
   - They are **merged** (`concatenate_4`) into a single `(None, 128)` vector.  
   - **Dense layers** (e.g., `dense_8`, `dense_9`) with Dropout in between refine the fused representation.  
   - **Output** (`Dense`) with 4 units (one for each precipitation category): shape `(None, 4)` and typically uses a softmax activation.

This multi-branch approach leverages both **time-series** (meteorological) and **spatiotemporal** (satellite) patterns to improve classification accuracy.

---

## Results

*(From `Notebook_1_Inference.ipynb`)*

### Baseline RNN (Notebook_4_Baseline_RNN_Model)
- **Accuracy**: ~66%
- **F1-Score**: ~0.64

### Hybrid Models

1. **24-Hour Window**  
   - *Automatic Weights (Notebook_5)*:  
     - Accuracy ~69%  
     - F1-Score ~0.67  
   - *Manual Weights (Notebook_6)*:  
     - Accuracy ~70%  
     - F1-Score ~0.68  

2. **48-Hour Window**  
   - *Automatic Weights (Notebook_7)*:  
     - Accuracy ~73%  
     - F1-Score ~0.72  
   - *Manual Weights (Notebook_8)*:  
     - Accuracy ~74%  
     - F1-Score ~0.73  

3. **72-Hour Window**  
   - *Automatic Weights + Additional 10 Epochs (Notebook_9)*:  
     - Accuracy ~76%  
     - F1-Score ~0.75  
   - *Automatic Weights (Notebook_10)*:  
     - Accuracy ~75%  
     - F1-Score ~0.74  
   - *Manual Weights + Additional 10 Epochs (Notebook_11)*:  
     - Accuracy ~77%  
     - F1-Score ~0.76  
   - *Manual Weights (Notebook_12)*:  
     - Accuracy ~76%  
     - F1-Score ~0.75  

**Overall**, the **72‐Hour Hybrid Model** with **Manual Weights** and **Additional 10 Epochs** (Notebook_11) yielded the **best performance**, with an accuracy of ~77% and F1-score of ~0.76. The confusion matrices across notebooks show this approach performs especially well in distinguishing **Medium Rain** vs. **High Rain**, where the baseline RNN struggled.

---

## Collaborators
- **Lokesh Balamurugan**  
- **Nitin Sai Varma Indukuri**  
- **Krishica Gopalakrishnan**  

---

## Contact
For questions, suggestions, or collaborations, please reach out:
- **Email**: youremail@example.com
- **GitHub**: [YourGitHubProfile](https://github.com/YourGitHubProfile)

> **Thank you** for exploring this project!
```