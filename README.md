# CIMAGE: Exploiting Conditional Independence in Masked Graph Auto-encoders

This repository contains the implementation of the paper presented at **CIKM 2024**: *"CIMAGE: Exploiting the Conditional Independence in Masked Graph Auto-encoders"*. 

## 📌 Dependencies
Please refer to the [`environment.yaml`](./environment.yaml) file for the required library dependencies.

## 🚀 Getting Started

### 📊 Node Classification
To train the model for node classification, run the following command:
```bash
python train_nodeclas.py --dataset photo --device 0
```
### 🔗 Link Prediction
To train the model for link prediction, use:
```bash
python train_linkpred.py --dataset cora --device 1
```
### 🔍 Reproducibility
Logs containing accuracy and loss metrics for each epoch are included to facilitate reproduction. If you encounter any challenges in reproducing the results, kindly refer to these logs for guidance.

