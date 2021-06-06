import numpy as np
import pandas as pd
import torch
import glob
import os
from os import path
import config
import pickle
from sklearn import metrics
from trainer.bert_model import BertIntentModel
import pymongo

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

def model_saver(le,max_len,state,filename):
    if not path.isdir(config.MODEL_PATH+'/'+'model_{}'.format(filename)):
        os.mkdir(config.MODEL_PATH+'/'+'model_{}'.format(filename))
    print("Storing Model in {}".format(config.MODEL_PATH+'/'+'model_{}'.format(filename)))
    torch.save(state, config.MODEL_PATH+'/'+'model_{}/model.pth'.format(filename))
    
    with open(config.MODEL_PATH+'/'+'model_{}/encoder.pkl'.format(filename), 'wb') as f:
        pickle.dump(le, f)
    
    with open(config.MODEL_PATH+'/'+'model_{}/max_len.pkl'.format(filename), 'wb') as f:
        pickle.dump({'max_len':max_len}, f)

    return None

def model_loader():
    try:
        latest_model_path = config.MODEL_PATH+'/model_'+str(max([int(os.path.split(model)[1].split('_')[1]) for model in glob.glob(config.MODEL_PATH+'/*')]))
        model_name = latest_model_path.split('/')[-1]
        print("Loading {}...".format(model_name))
        encoder = '{}/encoder.pkl'.format(latest_model_path)
        max_len = '{}/max_len.pkl'.format(latest_model_path)
        model_file = '{}/model.pth'.format(latest_model_path)
        with open(encoder, 'rb') as file:
            encoder = pickle.load(file)
            encoder = encoder.classes_

        with open(max_len, 'rb') as file:
            max_len = pickle.load(file)
            max_len = max_len['max_len']
    
        device = config.MODEL_CONFIG['bert']['device']
        tokenizer = config.MODEL_CONFIG['bert']['tokenizer']
        num_labels = len(encoder)
        model_config = config.MODEL_CONFIG['bert']['model_config']
        
        model = BertIntentModel(num_labels,model_config).to(device) 
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Model Loaded...")
        return model,encoder,tokenizer,max_len,model_name
    except:
        return None,None,None,None,None
    
    