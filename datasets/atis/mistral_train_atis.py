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
from datasets import Dataset, DatasetDict
import datasets

dataset_template = r'The conversation between Human and AI assistant. [INST] {} [\INST] {}'
sample_template = r'Text: "{}". Question: What is the intent of this text based on options: "Abbreviation", "Aircraft", "Airfare", "Airline", "Time", "Flight", "Quantity", "Ground Service"?'
label_template = r'Answer: {}'
dct_labels = {'0': 'Abbreviation', '1': 'Aircraft', '2': 'Airfare', '3': 'Airline', '4':'Time', '5':'Flight', '6':'Quantity', '7':'Ground Service'}

dct_rev_labels = {'Abbreviation':0, 'Aircraft':1, 'Airfare':2, 'Airline':3, 'Time':4, 'Flight':5, 'Quantity':6, 'Ground Service':7}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CURL_CA_BUNDLE'] = ''
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
new_model = 'Mistral-7B-atis'

def clean_last_char_if_not_alphabet(text):
    if not text[-1].isalpha():
        return text[:-1]
    else:
        return text

def clean_txt_res(text):
    try:
        tmp_arr = text.split()[1:3]
        final_str = None
        tmp_arr = [clean_last_char_if_not_alphabet(item) for item in tmp_arr]
        if tmp_arr[1] == 'positive' or tmp_arr[1] == 'negative':
            final_str = ' '.join(tmp_arr)
        else:
            final_str = tmp_arr[0]
        return final_str
    except:
        return None

def get_num_labels(text):
    try:
        return dct_rev_labels[text]
    except:
        return 0

for iteration in range(0,3):
    for model_type in ['gpt', 'gpt4', 'llama', 'play', 'mistral']:
        for method in ['prompt', 'taboo', 'hints', 'chaining', 'comb', 'taboo_ablt', 'hints_ablt', 'chaining_ablt']:
            for i in range(0,5):
                df = pd.read_csv('atis/'+model_type+'/'+str(i)+'/atis_'+method+'.csv')
                chat_samples = []
                labels = []
                for index, row in df.iterrows():
                    sample = sample_template.format(row['text'])
                    label = label_template.format(dct_labels[str(row['label'])])
                    labels.append(row['label'])
                    chat_sample = dataset_template.format(sample, label)
                    chat_samples.append(chat_sample)

                dct_df = {'chat_sample': chat_samples}
                df = pd.DataFrame.from_dict(dct_df).to_csv('atis/'+model_type+'/'+str(i)+'/train_atis_mistral_'+method+'.csv',index=False)

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )

                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.add_eos_token = True
                tokenizer.add_bos_token, tokenizer.add_eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    device_map="auto",
                    trust_remote_code=True,
                    load_in_4bit=True,
                    torch_dtype=torch.bfloat16,
                    quantization_config=bnb_config)

                model.config.use_cache = False # silence the warnings. Please re-enable for inference!
                model.config.pretraining_tp = 1
                model.gradient_checkpointing_enable()
                # Load tokenizer

                model = prepare_model_for_kbit_training(model)
                peft_config = LoraConfig(
                        r=16,
                        lora_alpha=16,
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
                    )
                model = get_peft_model(model, peft_config)

                df = pd.read_csv('atis/'+model_type+'/'+str(i)+'/train_atis_mistral_'+method+'.csv')
                dataset = Dataset.from_pandas(df)

                from datasets import load_dataset
                from trl import SFTTrainer
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer
                training_arguments = TrainingArguments(
                    output_dir= "./res_mist_atis",
                    num_train_epochs=20,
                    per_device_train_batch_size= 32,
                    gradient_accumulation_steps= 1,
                    optim = "paged_adamw_8bit",
                    save_steps=500,
                    logging_steps= 50,
                    learning_rate= 2e-5,
                    weight_decay= 0.01,
                    fp16= True,
                    bf16= False,
                    max_grad_norm= 0.3,
                    max_steps= -1,
                    warmup_ratio= 0.1,
                    group_by_length= True,
                    lr_scheduler_type= "constant"
                )
                # Setting sft parameters
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    peft_config=peft_config,
                    max_seq_length= 128,
                    dataset_text_field="chat_sample",
                    tokenizer=tokenizer,
                    args=training_arguments,
                    packing= False,
                )

                trainer.train()
                trainer.model.save_pretrained(new_model)
                wandb.finish()
                model.config.use_cache = True
                model.eval()

                generation_config = model.generation_config
                generation_config.temperature = 0.8
                generation_config.top_p = 0.9
                generation_config.pad_token_id = tokenizer.eos_token_id
                generation_config.eos_token_id = tokenizer.eos_token_id
                generation_config.num_return_sequences = 1
                generation_config.max_new_tokens = 10
                generation_config.do_sample = True

                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
                tokenizer.pad_token = tokenizer.eos_token

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

                def get_chat_samples(text, lbl):
                    dataset_template = r'The conversation between Human and AI assistant. [INST] {} [\INST]'
                    sample = sample_template.format(text)
                    chat_sample = dataset_template.format(sample)
                    return chat_sample

                df_test = pd.read_csv('atis/atis_test.csv').sample(frac=0.1)
                dataset_test = Dataset.from_pandas(df_test)
                df_test['chat_sample'] = df_test.apply(lambda x: get_chat_samples(x['text'], x['label']), axis=1)

                def get_text_result(prompt):
                    response =  request_response_from_falcon(prompt)[0]
                    loc_inst = response.find('[\\INST]')
                    sub_str = response[loc_inst+len('[\\INST]'):].strip()
                    return sub_str

                df_test['txt_pred'] = df_test['chat_sample'].apply(get_text_result)
                df_test.to_csv('atis/'+model_type+'/'+str(i)+'/mistral_ft_res_'+method+'_'+str(iteration)+'.csv',index=False)