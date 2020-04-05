from copy import deepcopy
import pandas as pd
from sklearn.metrics import roc_curve

ATTRIBUTE_SELECTION_DICT = dict()
ATTRIBUTE_SELECTION_DICT['CfsSubsetEval'] = ['PROPERTY TYPE',
                                            'ADDRESS',
                                            'CITY',
                                            'STATE OR PROVINCE',
                                            'ZIP OR POSTAL CODE']
ATTRIBUTE_SELECTION_DICT['OneRAttributeEval'] = ['SQUARE FEET', 
                                                 'LOCATION', 
                                                 'CITY', 'BATHS', 
                                                 'ZIP OR POSTAL CODE', 
                                                 'YEAR BUILT']
ATTRIBUTE_SELECTION_DICT['ClassifierAttributeEval'] = ['LONGITUDE',
                                                      'STATE OR PROVINCE',
                                                      'ZIP OR POSTAL CODE',
                                                      'CITY',
                                                      'BATHS']
ATTRIBUTE_SELECTION_DICT['InfoGainAttributeEval'] = ['ADDRESS',
                                                    'LOCATION',
                                                    'ZIP OR POSTAL CODE',
                                                    'CITY',
                                                    'BATHS']
ATTRIBUTE_SELECTION_DICT['Custom'] = ['PROPERTY TYPE',
                                     'ZIP OR POSTAL CODE',
                                     'BEDS',
                                     'BATHS',
                                     'SQUARE FEET',
                                     'LOT SIZE']

def one_hot_encode_col(df, x):
    dummies = pd.get_dummies(df[x], prefix=x, prefix_sep='_')
    new_df = pd.concat([df, dummies], axis=1).fillna(0)
    new_df.drop(x, inplace=True, axis=1)
    return new_df


def one_hot_encode_all_cols(df):
    new_df = deepcopy(df)
    
    for c in df.columns:
        if df[c].dtype not in ['float64', 'int64']:
            print("One hot encoding {}".format(c))
            new_df = one_hot_encode_col(new_df, c)
            
    return new_df


def run_all_attribute_selection_methods(X):
    d = {}
    for method, cols in ATTRIBUTE_SELECTION_DICT.items():
        print(method)
        X1 = X.loc[:, cols]
        X2 = one_hot_encode_all_cols(X1)
    d[method] = X2
    return d


def get_tpr_per_class(m):
    
    tprs = []
    for i in range(len(m)):
        tp = m[i][i]
        tp_plus_fn = sum(m[i])
        tpr = tp / tp_plus_fn
        tprs.append(tpr)
        
    return tprs

def plot_roc(y_bool, y_probas_max):
    fpr, tpr, thresholds = roc_curve(y_bool, y_probas_max)
    fig = plt.figure(figsize=(20, 15))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate', size=25)
    plt.ylabel('True Positive Rate', size=25)
    plt.title("ROC Curve")
    plt.show()



