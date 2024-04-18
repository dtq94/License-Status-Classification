import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import category_encoders as ce
import warnings
import warnings
warnings.filterwarnings("ignore")
# does target encoding for the column listed in ""

def target_encoding(df,col_to_transform):
    """This function target encodes the feature and returns the dataframe

    Args:
        df(pandas dataframe) : dataframe to be transformed  
        col_to_transform (List): list of column to be transformed 
    """
    enc=ce.OneHotEncoder().fit(df.target.astype(str))
    y_onehot=enc.transform(df.target.astype(str))
    class_names=y_onehot.columns
    for class_ in class_names:
        enc=ce.TargetEncoder(smoothing=0)
        temp = enc.fit_transform(df[col_to_transform],y_onehot[class_])
        temp.columns=[str(x)+'_'+str(class_) for x in temp.columns]
        df = pd.concat([df,temp],axis=1)
    return df

def random_sampling(df,target,target_prop):
    """Function to generate random sampling based on the target proportion

    Args:
        df(pandas dataframe) : dataframe to be transformed  
        target (List): Distinct value in the target column
        target_prop (List): Fraction of proportion to be present after the sampling process
    """
    df_list = []
    for i in range(len(target)):
        if target_prop[i] > 1:
            temp_df = df[df.target==target[i]].sample(frac=target_prop[i],replace=True)
        else:
            temp_df = df[df.target==target[i]].sample(frac=target_prop[i],replace=False)
        df_list.append(temp_df)
    final_df = pd.concat(df_list)
    return final_df

