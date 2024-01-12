import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import numpy as np
import itertools
import pickle

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset

from datasets import Dataset, DatasetDict

import argparse

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

def prepare_data(df):
    dataset = Dataset.from_pandas(df)
    tokenized_datasets = tokenizer(dataset["text"], padding=True, return_tensors='pt', truncation=True)
    tokenized_datasets['label'] = dataset['label']
    tokenized_datasets['text'] = dataset['text']
    ds = Dataset.from_dict(tokenized_datasets).with_format("torch")
    return ds
	
from transformers import BertForSequenceClassification
def get_new_model():
    model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path='bert-large-uncased',
            num_labels=5,
            classifier_dropout=0.2)
    return model
	
def get_formatted_structure_cart(guids, logits, golds, no_epoch):
    final_lst = []
    for guid, logit, gold in zip(guids, logits, golds):
        dct = {}
        dct['guid'] = guid
        dct['logits_epoch_'+str(no_epoch)] = list(logit)
        dct['gold'] = gold
        final_lst.append(dct)
    return final_lst
	
def eval_loop(model, ds):
    model.eval()
    test_loader = DataLoader(ds, batch_size=128, shuffle=False)
    with torch.no_grad():
        total_correct = 0
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            _, predicted = torch.max(outputs[1], 1)
            correct = (predicted == labels).sum().item()
            total_correct+=correct
    return (total_correct / len(ds)) * 100
	
from transformers import AdamW
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_loop(model, ds, BATCH_SIZE=64, NUM_EPOCHS=80):
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ds, batch_size=BATCH_SIZE*2, shuffle=False)
    model.to(device)
    
    acc_tests = []
    
    df_test = pd.read_csv('sst5/test_sst.csv')
    ds_test = prepare_data(df_test)

    history_cart_dct = {}

    optim = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0

        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

            total_loss += loss.item()
        
        if epoch % 10 == 0 and epoch != 0:
            print("EPOCH: " + str(epoch) + ", LOSS: " + str(total_loss/len(ds)))

        # eval and collect logits for the train dataset for cartography
#         model.eval()
#         with torch.no_grad():
#             guids = []
#             logits = []
#             golds = []

#             for batch in test_loader:
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 labels = batch['label'].to(device)
#                 outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#                 logits.extend(outputs[1].cpu().numpy())

#                 guids.extend(batch['text'])
#                 golds.extend(labels.cpu().numpy())

#             lst_hist = get_formatted_structure_cart(guids, logits, golds, epoch)
#             history_cart_dct[epoch] = lst_hist
            
        if epoch % 10 == 0 and epoch != 0:
            acc_test = eval_loop(model, ds_test)
            print("EPOCH: " + str(epoch) + ", ACC_TEST: " + str(acc_test))
            acc_tests.append(acc_test)
        
    return model, history_cart_dct, acc_tests

for model_name in ['mistral']:
    for method in ['taboo', 'chaining', 'hints']:
        for iteration in range(0,5):

            accs = []
            for i in range(0,5):
                df = pd.read_csv('sst5/'+model_name+'/'+str(iteration)+'/sst5_'+str(method)+'_ablt.csv')
                ds = prepare_data(df)

                model = get_new_model()
                model, history_cart_dct, acc_tests = train_loop(model, ds)
                accs.append(acc_tests)

                # with open('sst5/'+model_name+'/'+str(iteration)+'/history_'+str(method)+'_'+str(i)+'.pkl', 'wb') as handle:
                #     pickle.dump(history_cart_dct, handle)
                    
            with open('sst5/'+model_name+'/'+str(iteration)+'/'+str(method)+'_accs_ablt.pkl', 'wb') as handle:
                pickle.dump(accs, handle)