import torch
from flask import request
from flask_restful import Resource


class Predict(Resource):
    def __init__(self,collection,model,encoder,tokenizer,max_len,model_name,config):
        self.collection = collection
        self.model = model
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_len = max_len
        self.config = config

    def predict(self,utterance,model,encoder,tokenizer,max_len,model_name,config):
        model.eval()
        device = config.MODEL_CONFIG['bert']['device']
        inputs = tokenizer.encode_plus(
                    utterance,
                    None,
                    add_special_tokens=True,
                    max_length=max_len,
                    padding='max_length',
                    truncation=True,
                    return_token_type_ids=True
                )	

        ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device, dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device, dtype=torch.long)
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0).to(device, dtype=torch.long)	

        outputs = model(ids, mask, token_type_ids)
        softmax = torch.nn.Softmax(dim=1)
        
        probs = [round(p,3) for p in softmax(outputs)[0].cpu().detach().numpy().tolist()]
        
        _, preds = torch.max(outputs, dim=1)
        label = encoder[preds.item()]
        prob = probs[preds.item()]
        
        return {
            'intent':{
                'utterance':utterance,
                'name':label,
                'confidence':prob,
                'model_name':model_name,
            }
        }

    def post(self):
        utterance = request.json['utterance']
        if self.model is None:
            return {'message':"Please Train Model First"}
        else:
            return self.predict(utterance,self.model,self.encoder,self.tokenizer,self.max_len,self.model_name,self.config)
