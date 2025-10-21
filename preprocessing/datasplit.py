#retrieve the datasplit from multiindex file. hands over [ind_train, ind_val, ind_test, ind_holdout] 
import pandas as pd
def get_datasplit(df, ind_train, ind_val, ind_test, ind_holdout):
    """
    Retrieves the data split indices for training, validation, test, and holdout sets. Combines train and validation indices for final training.
    
    Parameters:
    df (pd.DataFrame): The complete dataset with a multi-index.
    ind_train (list): List of indices for the training set.
    ind_val (list): List of indices for the validation set.
    ind_test (list): List of indices for the test set.
    ind_holdout (list): List of indices for the holdout set.
    
    Returns:
    pd.DataFrame, pd.DataFrame, pd.DataFrame: DataFrames for training (combined train and val), test, and holdout sets.
    """
    #
    #get iris split from my cleaned data set
    df_train = df[df.set_index(["kinase", "ligand"]).index.isin(ind_train)].copy()
    df_val = df[df.set_index(["kinase", "ligand"]).index.isin(ind_val)].copy()
    df_test = df[df.set_index(["kinase", "ligand"]).index.isin(ind_test)].copy()
    df_holdout = df[df.set_index(["kinase", "ligand"]).index.isin(ind_holdout)].copy()
    #combine train and val for final training
    df_train = pd.merge(df_train, df_val, how="outer")   

    return [df_train, df_test, df_holdout]
