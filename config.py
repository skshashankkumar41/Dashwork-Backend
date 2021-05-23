import torch
from transformers import BertTokenizer, BertConfig

BASE_PATH = 'G:/AIP/Dashwork-Backend'
DATA_PATH = BASE_PATH + '/trainer/data'
MODEL_PATH = BASE_PATH + '/trainer/models'

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

    }
}
