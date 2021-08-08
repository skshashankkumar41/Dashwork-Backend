import torch
from transformers import BertTokenizer, BertConfig

BASE_PATH = ''
DATA_PATH = BASE_PATH + 'trainer/data'
MODEL_PATH = BASE_PATH + 'trainer/models/lstm'

MODEL_CONFIG = {
    'bert':{
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'lr':1e-4,
        'epochs':5,
        'pre_trained_model':"bert-base-uncased",
        'tokenizer': BertTokenizer.from_pretrained("bert-base-uncased"),
        "model_config":BertConfig(),
        'train_batch_size':4,
        'val_batch_size':4,

    },
    'lstm':{
        'model_path':BASE_PATH+'trainer/models/lstm',
        'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'lr':0.001,
        'epochs':15,
        'train_batch_size':6,
        'val_batch_size':3,

    }
}
