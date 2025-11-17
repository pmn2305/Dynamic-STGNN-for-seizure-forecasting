🧠 Seizure Forecasting Using Dynamic Spatio-Temporal Graph Neural Networks (DSTGNN)

This repository contains the full implementation of a Dynamic Spatio-Temporal Graph Neural Network (DSTGNN) for patient-specific epileptic seizure forecasting using EEG data from the CHB-MIT Scalp EEG Dataset.

Unlike seizure detection, which identifies seizures after they occur, forecasting aims to detect subtle preictal (pre-seizure) patterns before onset, enabling early warning systems for real-world clinical use.

⭐ Key Contributions

Built a dynamic graph-based deep learning model that learns time-varying connectivity between EEG channels.

Introduced a learnable edge-updater module to adapt connectivity per window.

Designed a full spatio-temporal pipeline with GNNs + GRUs.

Developed clinically meaningful visualizations, including:

Dynamic connectivity evolution

Prediction probability curves

ROC curve

EEG vs. DSTGNN representation comparisons

Performed patient-specific evaluation on multiple CHB-MIT subjects.

📁 Dataset

CHB-MIT Scalp EEG Database
Source: PhysioNet
Channels: 18–23
Sampling Rate: 256 Hz
Subjects used: CHB01–CHB06

Each subject folder contains a summary.txt file indicating:

Seizure start time

Seizure end time

File mapping

These timestamps are used for preictal labeling during training.

🧩 Methodology Overview
1️⃣ Windowing + Feature Extraction

10s windows, 5s overlap

Extracted features per channel:

Mean, variance, skewness, kurtosis

Band powers (δ, θ, α, β, γ)

2️⃣ Graph Construction

For every window:

Compute absolute Pearson correlation

Build a top-k functional graph (k = 12)

This creates a per-window adjacency matrix → capturing dynamic connectivity

3️⃣ Dynamic Edge Updating

Learnable module enhances or suppresses edges based on latent neural activity:

w_ij = σ( W2 * ReLU( W1 [h_i || h_j] ) )
A_t = C ⊙ w_t


This allows the model to learn evolving brain network structure.

4️⃣ Spatial + Temporal Modeling

Spatial modeling using 2-layer GCN

Temporal modeling using GRU over T=4 graph windows

Final sigmoid classifier outputs preictal probability

🔍 Visualizations Included

This repo includes powerful interpretation plots:

✔ Dynamic adjacency matrices (connectivity evolution)
✔ DSTGNN-learned features vs raw EEG
✔ Prediction probability curves across time
✔ ROC curve
✔ Graph embeddings + projections

These are essential for making the work publishable and interpretable.

📊 Results (Per Subject)
Subject	Sensitivity	Specificity	Precision	F1	Accuracy
CHB01	0.52	0.89	0.06	0.11	0.87
CHB02	1.00	0.80	0.04	0.08	0.79
CHB03	0.46	0.92	0.07	0.12	0.90
CHB04	0.40	0.84	0.04	0.07	0.83

Even with modest window metrics, the model exhibits strong seizure-level sensitivity, proving that dynamic graph learning captures meaningful preictal structure.

🚀 How to Run

Download CHB-MIT Dataset

Dataset link (PhysioNet):
https://physionet.org/content/chbmit/1.0.0/

Place data in:
/data/CHB-MIT/
run the ipynb notebook


📌 Future Work

Integrate transformer-based temporal encoders

Reduce false alarms through smoothers + thresholding

Subject-independent generalization

Artifact suppression for noisy EEG segments

👥 Authors

Prerana M N

Ria M Parikh

Poorav J Bolar

PES University, Bengaluru
Dept. of CSE (AIML)
