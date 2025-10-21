#Feature Scaling functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def scale_features(train_features, val_features, test_features):
    """
    Scales the features using Min-Max scaling based on the training set.
    
    Parameters:
    train_features (pd.DataFrame): Training feature set.
    val_features (pd.DataFrame): Validation feature set.
    test_features (pd.DataFrame): Test feature set.
    
    Returns:
    pd.DataFrame, pd.DataFrame, pd.DataFrame: Scaled training, validation, and test feature sets.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_features)
    
    train_scaled = pd.DataFrame(scaler.transform(train_features), columns=train_features.columns, index=train_features.index)
    val_scaled = pd.DataFrame(scaler.transform(val_features), columns=val_features.columns, index=val_features.index)
    test_scaled = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns, index=test_features.index)
    
    #tranforms them to tensors for pytorch
    

    train_tensor = torch.tensor(train_scaled.values, dtype=torch.float32)
    val_tensor = torch.tensor(val_scaled.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_scaled.values, dtype=torch.float32)

    return train_tensor, val_tensor, test_tensor