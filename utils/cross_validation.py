import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, KFold

def convert_to_array(X):
        if (isinstance(X, pd.DataFrame) or (isinstance(X, pd.Series))):
            X = X.get_values()
        return X
    
def split_big(arr, thrs):
    size = len(np.concatenate(arr))
    narr = []
    for a in arr:
        if (len(a)/size > thrs):
            split = max(2, len(a)/(size*thrs))
            splitted = np.array_split(a, split)
            narr.append(splitted[0])
            narr.append(splitted[1])
        else:
            narr.append(a)
    return np.array(narr)
        
def cross_val_by(data, by, test_train_split, delete_by_column, cross_val_shuffle = True, random_state = 23):
    if (isinstance(data[0], pd.DataFrame)):
        datacolumns = data[0].columns.get_values()
        if (isinstance(by, str)):
            by = np.where(datacolumns==by)[0][0]
        data[0] = convert_to_array(data[0])
        data[1] = convert_to_array(data[1])
    by_column = np.array([d[by] for d in data[0]])
    indexes = [np.where(by_column == s)[0] for s in np.unique(by_column)]
        
    if (delete_by_column):
        data[0] = np.delete(data[0], by, axis=1)
        
    test_size = 1./test_train_split
    indexes = split_big(indexes, test_size)
    splits = []
    kfold = KFold(len(indexes), int(test_train_split), cross_val_shuffle, random_state)
    for train, test in kfold:
        train_ind = np.concatenate(indexes[train])
        test_ind = np.concatenate(indexes[test])
        splits.append([train_ind, test_ind])
    return splits