# CrnnCrispr
An interpretable deep learning method for CRISPR/Cas9 sgRNA on-target activity prediction.    
Below is the layout of the whole model.    
![image](https://github.com/White-Cement/CrnnCrispr/assets/104978990/787d4e04-8ae5-4ec7-b527-e78c36a646db)
# Environment
* Keras 2.4.3
* TensorFlow-GPU 2.5.0
* cudatoolkit 11.0
# Datasets
Include 10 public datasets:
* Benchmark
* WT
* ESP
* HF
* xCas
* SpCas9
* Sniper
* HCT116
* HELA
* HL60
# File description
* model.py: The CrnnCrispr model with CNN and RNN.
* model_train.py: Running this file to train the CrnnCrispr model. (5-fold cross-validation)
* model_test.py: Running this file to evaluate the CrnnCrispr model. (Demonstrate model performance by evaluating metrics through two regression question evaluation indicators)
