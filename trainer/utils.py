import numpy as np
import pandas as pd
import torch
import os
from os import path
import config
import pickle
from sklearn import metrics
import pymongo

def db_connector():
    client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')
    db = client['dashwork']
    intent_collection = db.intent
    return intent_collection

def explode(df, lst_cols, fill_value='', preserve_index=False):
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    idx_cols = df.columns.difference(lst_cols)
    lens = df[lst_cols[0]].str.len()
    idx = np.repeat(df.index.values, lens)
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    if (lens == 0).any():
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    res = res.sort_index()
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res

def print_metrics(true, pred, loss, type):
    accuracy = metrics.accuracy_score(true,pred)
    f1_score_micro = metrics.f1_score(true, pred, average='micro',zero_division = 1)
    print("-------{} Evaluation--------".format(type))
    print("CE Loss: {:.4f}".format(loss))
    print("Accuracy: {:.4f}".format(accuracy))
    print("F1-measure Micro: {:.4f}".format(f1_score_micro))
    print("------------------------------------")
    # return accuracy,loss
    return accuracy, loss 

def model_saver(le,state,filename):
    if not path.isdir(config.MODEL_PATH+'/'+'model_{}'.format(filename)):
        os.mkdir(config.MODEL_PATH+'/'+'model_{}'.format(filename))
    save_path = ''
    torch.save(state, config.MODEL_PATH+'/'+'model_{}/model.pth'.format(filename))
    
    with open(config.MODEL_PATH+'/'+'model_{}/encoder.pkl'.format(filename), 'wb') as f:
        pickle.dump(le, f)

    return None