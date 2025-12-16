# machine-learning-models-for-kinetic-rates-prediction

# Master Thesis Project: Machine learning models to predict kinetic rates of kinases 

All self-written code with exceptions: "feature_calc" and "dataset_operations", which were provided by Group Member Iris Guo

## Requirements:
- Python 3.12.7
- torch installed
- rdkit package 2025.9.1
- mordredcommunity 2.0.6
- propy3 1.1.1

## Build the Database: 
- run preprocessing/Feature_Generation.ipynb

## Train and Test Models

- run jupyter notebookes in the model folder

NN (Neural Network models):

- NN1 for the ligand feature variatiolns
- NN2 for an automated protein feature variation across all available Propy3 descriptors
- 1 hidden NN and 3 hidden NN, models to asses impact of complexity for predictiove performanca
- Hyperparameter tuning, based on the best R^2 on validation attained from NN1 and NN2

GNN (Graph Neural Network model)
