import pandas as pd
import re
import matplotlib.pyplot as plt
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


from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset

from datasets import Dataset, DatasetDict

import argparse

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-i", "--Iteration", help = "Iteration")
#parser.add_argument("-m", "--Method", help = "Diversity method")
 
# Read arguments from command line
args = parser.parse_args()

iteration = args.Iteration
#method = args.Method

df = pd.read_csv('atis/gpt/'+str(iteration)+'/atis_chaining.csv')

dataset = Dataset.from_pandas(df)

#dataset = load_dataset("csv", data_files="fb_llama/full_fb_taboo_llama", split='train', keep_default_na=False)

#dct_dataset = dataset.train_test_split(test_size=0.1)

#dataset_train = dct_dataset['train']
#dataset_test = dct_dataset['test']

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
#tokenized_test_datasets = dataset_test.map(tokenize_function, batched=True)

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path='bert-large-uncased',
        num_labels=8,
        attention_probs_dropout_prob=0.2,
        hidden_dropout_prob=0.2,
        classifier_dropout=0.2)

#from transformers import AutoModelForSequenceClassification

#model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=10)

from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="train_sst")

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer
test_dataset = load_dataset("csv", data_files="atis/atis_test.csv", split='train', keep_default_na=False)

training_args = TrainingArguments(output_dir="train_sst", evaluation_strategy="no", per_device_eval_batch_size=8, per_device_train_batch_size=4, warmup_steps=10, learning_rate=2e-5, num_train_epochs=20, save_steps=3000)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    compute_metrics=compute_metrics
)

trainer.train()

tokenized_dataset_all_gpt = test_dataset.map(tokenize_function, batched=True)
print(str(trainer.evaluate(tokenized_dataset_all_gpt)))
