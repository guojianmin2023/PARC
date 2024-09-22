# **Project Name**: Parameter-aware Reservoir Computing

## **Features**
We use the parameter-aware RC method to predict the time series, return map, and bifurcation diagram of the Logistic map. We also use the same method to predict the time series, phase diagram, and bifurcation diagram of the Chua circuit. We demonstrate that for Reservoirs of the same size with the same system, after training with the same method, the prediction accuracy is affected by the number and position of the bifurcation parameters in the training set.

## **Table of Contents**
- [Introduction](#introduction)
- [Logistic Map Prediction](#logistic-map-prediction)
- [Chua Circuit Bifurcation Prediction](#chua-circuit-bifurcation-prediction)
- [Training Process](#training-process)
- [Testing and Plotting](#testing-and-plotting)

## **Introduction**
This project implements key results in bifurcation prediction using parameter-aware reservoir computing (RC). The main goal is to reconstruct bifurcation diagrams for the Logistic map and Chua circuit using a parameter-aware method.

## **Logistic Map Prediction**
The code for Fig 3(a) is located in the `predicted logistic map` folder. In this folder:
- `data_logistic_dif4a.m`: Generates training data (`traindata.mat`).
- `predicted_log_bif.m`: The main script, which reconstructs the Logistic map bifurcation using the parameter-aware RC method.

## **Chua Circuit Bifurcation Prediction**
The code for Fig 5(a) is in the `predicted bifurcation of chua circuit` folder. It uses the parameter-aware RC method to reconstruct the Chua circuit bifurcation with four bifurcation parameter sample points.

## **Training Process**
The training process for optimizing hyperparameters is implemented in the `train` folder:
- `opt_attractor_with_lable.m`: The main script for optimizing hyperparameters. Before running this script, you need to place two empty matrices named `min_rng_set.mat` and `min_rmse_dynamic_set.mat` in the same folder. These matrix names can be modified as per your requirements.
- `func_train_attractor_with_lable.m`: The function used for training the RC model.

## **Testing and Plotting**
In the `predict and plot` folder:
- `chua_bif_predict_main_test.m`: The main script for reconstructing the Chua circuit bifurcation.
  - The matrix `bif_chua_pre5.mat` contains the data for the reconstructed bifurcation diagram.
  - `traindata.mat`: The training data matrix.
  - `opt_attractor_2_20240716T172246_814.mat` and `min_rng_set.mat`: Optimized data matrices.

