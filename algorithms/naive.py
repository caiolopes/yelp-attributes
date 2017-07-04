import pandas as pd
import numpy as np

def naive(data, query, classIndex = -1):
    class_col_name = data.columns[classIndex]
    classes = list(data[class_col_name].unique())
    classes_prob = np.zeros(len(classes))
    j = 0
    cols = list(data.columns)
    del cols[classIndex]
    for i in range(len(classes)):
        prob = 1
        for item in query:
            if not pd.isnull(item):
                data_slice = data[data[cols[j]] == item][class_col_name]
                prob = prob * (data_slice.value_counts()[classes[i]]/data[data[class_col_name] == classes[i]].shape[0])
            j = j + 1
        j = 0
        classes_prob[i] = prob*data[data[class_col_name] == classes[i]].shape[0]/data.shape[0]
    classes_prob = classes_prob/sum(classes_prob)
    classes_prob = [classes, classes_prob]

    return classes_prob

