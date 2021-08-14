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
from trainer.lstm_model import LSTMIntentModel
from trainer.transformer_model import TransformerIntentModel
from trainer.custom_tokenizer import Vocabulary
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

def print_metrics(epoch,train_acc,train_f1_score,train_loss,val_acc,val_f1_score,val_loss,max_epoch,run_time):
    run_time = round(run_time,1)
    print(("Epoch: {}/{}  {}s  train_loss: {:.4f}  train_acc: {:.2f}  train_f1: {:.2f}  eval_loss: {:.2f}  eval_acc:{:.2f}  val_f1: {:.2f} "
                     .format(epoch,max_epoch,run_time,train_loss, train_acc,train_f1_score, val_loss, val_acc,val_f1_score)))
    return None

def evaluater(true,pred,loss):
    accuracy = metrics.accuracy_score(true,pred)
    f1_score_micro = metrics.f1_score(true, pred, average='micro',zero_division = 1)
    return accuracy, f1_score_micro, loss 

def model_saver(le,state,filename,max_len=None,vocab=None):
    if not path.isdir('{}'.format(filename)):
        os.mkdir('{}'.format(filename))
    print("Storing Model in {}\n".format('{}'.format(filename)))
    torch.save(state, '{}/model.pth'.format(filename))
    
    with open('{}/encoder.pkl'.format(filename), 'wb') as f:
        pickle.dump(le, f)

    with open('{}/max_len.pkl'.format(filename), 'wb') as f:
        pickle.dump({'max_len':max_len}, f)
    
    with open('{}/vocab.pkl'.format(filename), 'wb') as f:
        pickle.dump(vocab, f)

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

def lstm_model_loader():
    try:
        latest_model_path = config.MODEL_CONFIG['lstm']+'/model_'+str(max([int(os.path.split(model)[1].split('_')[1]) for model in glob.glob(config.MODEL_PATH+'/*')]))
        model_name = latest_model_path.split('/')[-1]
        print("Loading {}...".format(model_name))
        encoder = '{}/encoder.pkl'.format(latest_model_path)
        model_file = '{}/model.pth'.format(latest_model_path)
        vocab = '{}/vocab.pkl'.format(latest_model_path)
        max_len = '{}/max_len.pkl'.format(latest_model_path)

        with open(encoder, 'rb') as file:
            encoder = pickle.load(file)
            encoder = encoder.classes_

        with open(vocab, 'rb') as file:
            vocab = pickle.load(file)

        with open(max_len, 'rb') as file:
            max_len = pickle.load(file)
            max_len = max_len['max_len']

        device = config.MODEL_CONFIG['lstm']['device']
        num_labels = len(encoder)

        model = LSTMIntentModel(vocab,num_labels,infer=True).to(device)
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Model Loaded...")
        return model,encoder,vocab,max_len,model_name

    except:
        return None,None,None,None,None

def transformer_model_loader():
    try:
        latest_model_path = config.MODEL_CONFIG['transformer']+'/model_'+str(max([int(os.path.split(model)[1].split('_')[1]) for model in glob.glob(config.MODEL_PATH+'/*')]))
        model_name = latest_model_path.split('/')[-1]
        print("Loading {}...".format(model_name))
        encoder = '{}/encoder.pkl'.format(latest_model_path)
        model_file = '{}/model.pth'.format(latest_model_path)
        vocab = '{}/vocab.pkl'.format(latest_model_path)
        max_len = '{}/max_len.pkl'.format(latest_model_path)

        with open(encoder, 'rb') as file:
            encoder = pickle.load(file)
            encoder = encoder.classes_

        with open(vocab, 'rb') as file:
            vocab = pickle.load(file)

        with open(max_len, 'rb') as file:
            max_len = pickle.load(file)
            max_len = max_len['max_len']

        device = config.MODEL_CONFIG['transformer']['device']
        num_labels = len(encoder)

        model = TransformerIntentModel(vocab,num_labels,infer=True).to(device)
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Model Loaded...")
        return model,encoder,vocab,max_len,model_name

    except:
        return None,None,None,None,None
    