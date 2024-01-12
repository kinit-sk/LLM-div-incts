import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
import pickle
from datasets import load_dataset
import pandas as pd
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['CURL_CA_BUNDLE'] = ''

MODEL_NAME = "garage-bAInd/Platypus2-70B-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","up_proj","o_proj","k_proj","down_proj","gate_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

generation_config = model.generation_config
generation_config.temperature = 0.8
generation_config.top_p = 0.9
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
generation_config.num_return_sequences = 1
generation_config.max_new_tokens = 200
generation_config.do_sample = True

df_train = pd.read_csv('20news/train_news.csv')


def change_label_except_for(x, label):
    if x == label:
        return 1
    else:
        return 0
    
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
def cleantext(string):
    text = string.split()
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

def get_fit_svm_linear_and_count_vectorizer(df_orig):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_orig['text'])
    x=X.toarray()
    y=df_orig['label']
    model=svm.SVC(kernel='linear')
    model.fit(x,y)
    return model, vectorizer, X

def get_freqs(vectorizer, X):
    feature_names = vectorizer.get_feature_names_out()
    dct = {'word': [], 'freq': []}
    for freq, word in zip(np.asarray(X.sum(axis=0))[0], feature_names):
        dct['word'].append(word)
        dct['freq'].append(freq)

    return pd.DataFrame.from_dict(dct)

def get_coefficients(model, vectorizer):
    feature_names = vectorizer.get_feature_names_out() 
    coefs_with_fns = sorted(zip(model.coef_[0], feature_names)) 
    df=pd.DataFrame(coefs_with_fns)
    df.columns='coefficient','word'
    return df.sort_values(by='coefficient')

import spacy
from spacy import displacy

def get_taboo_w_for_df_no_ner(df_orig, no_taboo_w, seed_samples_dct):
    dct_taboo_w_per_label = {}

    NER = spacy.load("en_core_web_sm")

    # we gather taboowords for each label in a one (desired label) vs. one (other labels) setting
    labels = list(set(df_orig['label']))
    for label in labels:
        # reset dataframe
        sub_df_orig = df_orig.copy()
        sub_df_orig['text'] = sub_df_orig['text'].map(lambda x: cleantext(x))
        # set setting to one vs one
        sub_df_orig['label'] = sub_df_orig['label'].map(lambda x: change_label_except_for(x, label))

        model, vectorizer, X = get_fit_svm_linear_and_count_vectorizer(sub_df_orig)
        freqs = get_freqs(vectorizer, X)
        coeffs = get_coefficients(model, vectorizer)
        
        sents = seed_samples_dct[label]
        ners = set()
        for sent in sents:
            res = NER(sent.lower())
            for txt in res.ents:
                subs = set(txt.text.lower().split())
                ners = ners.union(subs)
            
        joined = coeffs.set_index('word').join(freqs.set_index('word'), lsuffix='_caller', rsuffix='_other')
        joined_rel = joined[joined['freq'] >= 5].sort_values(by=['freq'])

        joined_rel = joined_rel.sort_values(by=['coefficient'])
        taboo_w_without_ints = list(joined_rel.index)
        taboo_w_without_ints = list(filter(lambda word: not word.isdigit(), taboo_w_without_ints))
        
        taboo_w_without_ners = list(filter(lambda word: not word in ners, taboo_w_without_ints))
        
        dct_taboo_w_per_label[label] = taboo_w_without_ners[-no_taboo_w:]
        
    return dct_taboo_w_per_label
    
    
from sentence_transformers import SentenceTransformer
sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_embs_for_sents(df_pd) -> dict:
    sents_dct = {}
    emb_dct = {}

    for dct in df_pd.to_dict('records'):
        if dct['label'] in sents_dct:
            sents_dct[dct['label']].append(dct['text'])
        else:
            sents_dct[dct['label']] = [dct['text']]
            
    # TODO: check if order is same
    for label in sents_dct.keys():
        emb_dct[label] = {'emb': sent_model.encode(sents_dct[label]), 'sent': sents_dct[label]}
    return emb_dct

def calculate_outliers(df_pd) -> dict:
    embs_dct = get_embs_for_sents(df_pd)
    mean_dct = {}
    pandas_dct = {'label': [], 'distance': [], 'text': []}
    
    # calculate mean vector per label
    for label in embs_dct:
        mean_dct[label] = embs_dct[label]['emb'].mean(axis=0)
        
    # calculate distance from the mean vector per label
    for label in embs_dct:
        mean_emb = mean_dct[label]
        for (sent_emb, sent) in zip(embs_dct[label]['emb'], embs_dct[label]['sent']):
            dist = np.linalg.norm(mean_emb - sent_emb)
            pandas_dct['label'].append(label)
            pandas_dct['distance'].append(dist)
            pandas_dct['text'].append(sent)                        
    return pd.DataFrame.from_dict(pandas_dct)

def get_seed_sentences_per_labels(outliers_df, dct_phrases: dict) -> dict:
    dct_seeds_per_label = {}
    for label in dct_phrases.keys():
        no_samples = len(dct_phrases[label])
        sub_outlier_df = outliers_df[outliers_df['label'] == label].sort_values(by=['distance'], ascending=False)
        dct_seeds_per_label[label] = list(sub_outlier_df.head(no_samples)['text'])
    return dct_seeds_per_label
    
    
import re
import string

def filter_responses(dct_responses):
    dct_df = {'label': [], 'text': [], 'seed': []}
    for key in dct_responses:
        for responses in dct_responses[key]:
            for response in responses[0]:
                loc_inst = responses[0][0].find('Response:')
                sub_str = responses[0][0][loc_inst:].strip()
                contents = sub_str.split('\n')[1:]

                for content in contents:
                    content = content.strip()
                    if len(content) == 0  or content.isspace():
                        continue
                    if content[0] == '1' or content[0] == '2' or content[0] == '3':
                        content = content[3:]
                    if 'paraphr' in content:
                        continue
                    dct_df['label'].append(key)
                    dct_df['text'].append(content)
                    dct_df['seed'].append(responses[1])

    fb_0 = pd.DataFrame.from_dict(dct_df)

    fb_0['text']=fb_0['text'].apply(lambda x: x.lower())
    fb_0['text']=fb_0['text'].apply(lambda x: x.strip())
    fb_0['text']=fb_0['text'].apply(lambda x: x.replace('"',''))
    fb_0['text']=fb_0['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

    fb_0 = fb_0.drop_duplicates().dropna()
        
    return fb_0

# def llama_v2_prompt(
#     messages: list[dict]
# ):
#     msgs = [{'role': 'user', 'content': messages[0]['content']}]

#     return tokenizer.apply_chat_template(msgs, tokenize=False)
    
def llama_v2_prompt(messages: list[dict]):
    DEFAULT_SYSTEM_PROMPT = """### Instruction:

    {}

    ### Response:
    """

    return DEFAULT_SYSTEM_PROMPT.format(messages[0]['content'])
    

def collect_samples(dct_final_prompts):
    dct_responses = {}

    for idx, key in enumerate(dct_final_prompts):
        print(str(idx))
        dct_responses[key] = []
        for prompt in dct_final_prompts[key]:
            messages = [{"role": "user", "content": prompt[0]}]
            str_message = llama_v2_prompt(messages)
            responses = request_response_from_falcon(str_message)
            dct_responses[key].append((responses, prompt[1]))
            
    return dct_responses

		
def request_response_from_falcon(prompt):
    device = "cuda:0"
    outputs_sent = []
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
      outputs = model.generate(
          input_ids = encoding.input_ids,
          attention_mask = encoding.attention_mask,
          generation_config = generation_config
      )
      for output in outputs:
        outputs_sent.append(tokenizer.decode(output, skip_special_tokens=True))
    return outputs_sent

for NO_TRY in range(0,2):
    N_SAMPLES = 6
    get_subsampled = df_train.groupby('label', group_keys=False).apply(lambda x: x.sample(N_SAMPLES))

    get_subsampled.to_csv('20news/play/'+str(NO_TRY)+'/seeds.csv', index=False)
        
    dct_phrases = {}
    for key in set(get_subsampled['label']):
        dct_phrases[key] = list(get_subsampled[get_subsampled['label'] == key]['text'])

    default_prompt = 'Rephrase an original question or statement 3 times. Original phrase: "{}".'

    dct_final_prompts = {}

    for key in dct_phrases:
        dct_final_prompts[key] = []
        for phrase in dct_phrases[key]:
            dct_final_prompts[key].append((default_prompt.format(phrase), phrase))
            
    dct_responses = collect_samples(dct_final_prompts)

    fb_0 = filter_responses(dct_responses)
    fb_0.drop_duplicates().to_csv('20news/play/'+str(NO_TRY)+'/fb_0_seeds.csv', index=False)

    dct_responses = collect_samples(dct_final_prompts)

    fb_0 = filter_responses(dct_responses)
    fb_0.drop_duplicates().to_csv('20news/play/'+str(NO_TRY)+'/fb_1_seeds.csv', index=False)

    fb_0 = pd.read_csv('20news/play/'+str(NO_TRY)+'/fb_0_seeds.csv')
    fb_0['text'] = fb_0['text'].astype("string")

    fb_0['text'] = fb_0['text'].astype("string")

    dct_phrases = fb_0.groupby('label')['seed'].apply(set).to_dict()
    dct_taboo = get_taboo_w_for_df_no_ner(fb_0, 3, dct_phrases)

    defaul_taboo_prompt = 'Rephrase an original question or statement 3 times. Original phrase: "{}". Don’t use the words “{}”, “{}” or “{}” in your responses.'

    dct_final_prompts = {}

    for key in dct_phrases:
        dct_final_prompts[key] = []
        for phrase in dct_phrases[key]:
            dct_final_prompts[key].append((defaul_taboo_prompt.format(phrase, dct_taboo[key][0], dct_taboo[key][1], dct_taboo[key][2]), phrase))
            
    dct_responses = collect_samples(dct_final_prompts)

    fb_0 = filter_responses(dct_responses)
    fb_0.drop_duplicates().to_csv('20news/play/'+str(NO_TRY)+'/fb_1_taboo.csv', index=False)

    fb_0 = pd.read_csv('20news/play/'+str(NO_TRY)+'/fb_0_seeds.csv').drop_duplicates()
    fb_0['text'] = fb_0['text'].astype("string")
    dct_phrases = fb_0.groupby('label')['seed'].apply(set).to_dict()

    df_outliers = calculate_outliers(fb_0)
    dct_phrases = get_seed_sentences_per_labels(df_outliers, dct_phrases)

    default_prompt = 'Rephrase an original question or statement 3 times. Original phrase: "{}".'

    dct_final_prompts = {}

    for key in dct_phrases:
        dct_final_prompts[key] = []
        for phrase in dct_phrases[key]:
            dct_final_prompts[key].append((default_prompt.format(phrase), phrase))
            
    dct_responses = collect_samples(dct_final_prompts)

    fb_0 = filter_responses(dct_responses)
    fb_0.drop_duplicates().to_csv('20news/play/'+str(NO_TRY)+'/fb_1_chaining.csv', index=False)

    default_prompt = """Rephrase an original question or statement 3 times. Original phrase: "{}".
    Example paraphrases:
    {}
    """

    default_hint_prompt = '"{}".'

    def get_hint_sentences_per_labels(df_outliers, no_samples, dct_phrases):
        dct_hints_per_sample = {}
        for label in dct_phrases.keys():
            for phrase in dct_phrases[label]:
                sub_df = df_outliers[df_outliers['seed'] == phrase] 
                sub_df = sub_df.sort_values(by=['distance'], ascending=False)
                dct_hints_per_sample[phrase] = list(sub_df.head(no_samples)['text'])
        return dct_hints_per_sample

    fb_0 = pd.read_csv('20news/play/'+str(NO_TRY)+'/fb_0_seeds.csv').drop_duplicates()
    fb_0['text'] = fb_0['text'].astype("string")
    dct_phrases = fb_0.groupby('label')['seed'].apply(set).to_dict()

    df_outliers = calculate_outliers(fb_0)

    df_merged = df_outliers.merge(fb_0, how='inner', on='text').drop_duplicates()[['label_x', 'text', 'distance', 'seed']]

    df_merged = df_merged.rename(columns={'label_x': 'label'})

    dct_hints_per_sample = get_hint_sentences_per_labels(df_merged, 3, dct_phrases)

    dct_final_prompts = {}

    for key in dct_phrases:
        dct_final_prompts[key] = []
        for phrase in dct_phrases[key]:
            hints = dct_hints_per_sample[phrase]
            str_hints = []
            for hint in hints:
                str_hints.append(default_hint_prompt.format(hint))
            final_hint_str = "\n".join(str_hints) 
            dct_final_prompts[key].append((default_prompt.format(phrase, final_hint_str), phrase))
            
    dct_responses = collect_samples(dct_final_prompts)

    fb_0 = filter_responses(dct_responses)
    fb_0.drop_duplicates().to_csv('20news/play/'+str(NO_TRY)+'/fb_1_hints.csv', index=False)
