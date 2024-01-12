import pandas as pd
from nltk.tree import Tree
from zss import simple_distance, Node
import pickle

# NOTE: This runs a very long time (5-12h)!

NO_TRY = 4

file = open('atis/gpt/'+str(NO_TRY)+'/syntax_prompt.pkl','rb')
(dct_parse_trees_from_stan, dct_parse_trees_final) = pickle.load(file)

df = pd.read_csv('atis/gpt/'+str(NO_TRY)+'/atis_prompt.csv').dropna()

dct_intents_with_txts = {}
for index, row in df.iterrows():
    if row['label'] in dct_intents_with_txts:
        dct_intents_with_txts[row['label']].append(row['text'])
    else:
        dct_intents_with_txts[row['label']] = [row['text']]

dct_results_based_intent = {}

for intent in dct_intents_with_txts.keys():
    if intent in dct_results_based_intent:
        continue
    res_lst = []
    lst_seed = dct_intents_with_txts[intent]
    lst_traverse = dct_intents_with_txts[intent]
    print(intent)
    
    for idx, seed_txt in enumerate(lst_seed):
        if idx % 10 == 0:
            print(idx)
            
        try:    
            parse_seed = dct_parse_trees_from_stan[seed_txt]
            
            seed_tree = dct_parse_trees_final[seed_txt]

            idx_trav = idx + 1
        
        except:
            idx_trav = idx + 1
            continue
        
        while (idx_trav < len(lst_traverse)):
            trav_txt = lst_traverse[idx_trav]
            
            # parse_trav = next(parser.raw_parse(trav_txt))
            try:
                parse_trav = dct_parse_trees_from_stan[trav_txt]
                
                trav_tree = dct_parse_trees_final[trav_txt]
            
#             trav_tree = Node(parse_trav.label())
#             trav_tree = assign_children(trav_tree, parse_trav[0])
            
                res = simple_distance(seed_tree, trav_tree)
                res_lst.append(res)
            except:
                idx_trav += 1
                continue
            
            idx_trav += 1
        
    dct_results_based_intent[intent] = res_lst
    
file = open('atis/gpt/'+str(NO_TRY)+'/syntax_prompt_res.pkl', 'wb')
# dump information to that file
pickle.dump(dct_results_based_intent, file)
# close the file
file.close()

file = open('atis/gpt/'+str(NO_TRY)+'/syntax_taboo.pkl','rb')
(dct_parse_trees_from_stan, dct_parse_trees_final) = pickle.load(file)

df = pd.read_csv('atis/gpt/'+str(NO_TRY)+'/atis_taboo.csv').dropna()

dct_intents_with_txts = {}
for index, row in df.iterrows():
    if row['label'] in dct_intents_with_txts:
        dct_intents_with_txts[row['label']].append(row['text'])
    else:
        dct_intents_with_txts[row['label']] = [row['text']]

dct_results_based_intent = {}

for intent in dct_intents_with_txts.keys():
    if intent in dct_results_based_intent:
        continue
    res_lst = []
    lst_seed = dct_intents_with_txts[intent]
    lst_traverse = dct_intents_with_txts[intent]
    print(intent)
    
    for idx, seed_txt in enumerate(lst_seed):
        if idx % 10 == 0:
            print(idx)
            
        try:    
            parse_seed = dct_parse_trees_from_stan[seed_txt]
            
            seed_tree = dct_parse_trees_final[seed_txt]

            idx_trav = idx + 1
        
        except:
            idx_trav = idx + 1
            continue
        
        while (idx_trav < len(lst_traverse)):
            trav_txt = lst_traverse[idx_trav]
            
            try:
                parse_trav = dct_parse_trees_from_stan[trav_txt]
                
                trav_tree = dct_parse_trees_final[trav_txt]
            
#             trav_tree = Node(parse_trav.label())
#             trav_tree = assign_children(trav_tree, parse_trav[0])
            
                res = simple_distance(seed_tree, trav_tree)
                res_lst.append(res)
            except:
                idx_trav += 1
                continue
            
            idx_trav += 1
        
    dct_results_based_intent[intent] = res_lst
    
file = open('atis/gpt/'+str(NO_TRY)+'/syntax_taboo_res.pkl', 'wb')
# dump information to that file
pickle.dump(dct_results_based_intent, file)
# close the file
file.close()

file = open('atis/gpt/'+str(NO_TRY)+'/syntax_chaining.pkl','rb')
(dct_parse_trees_from_stan, dct_parse_trees_final) = pickle.load(file)

df = pd.read_csv('atis/gpt/'+str(NO_TRY)+'/atis_chaining.csv').dropna()

dct_intents_with_txts = {}
for index, row in df.iterrows():
    if row['label'] in dct_intents_with_txts:
        dct_intents_with_txts[row['label']].append(row['text'])
    else:
        dct_intents_with_txts[row['label']] = [row['text']]

dct_results_based_intent = {}

for intent in dct_intents_with_txts.keys():
    if intent in dct_results_based_intent:
        continue
    res_lst = []
    lst_seed = dct_intents_with_txts[intent]
    lst_traverse = dct_intents_with_txts[intent]
    print(intent)
    
    for idx, seed_txt in enumerate(lst_seed):
        if idx % 10 == 0:
            print(idx)
            
        try:    
            parse_seed = dct_parse_trees_from_stan[seed_txt]
            
            seed_tree = dct_parse_trees_final[seed_txt]

            idx_trav = idx + 1
        
        except:
            idx_trav = idx + 1
            continue
        
        while (idx_trav < len(lst_traverse)):
            trav_txt = lst_traverse[idx_trav]
            
            try:
                parse_trav = dct_parse_trees_from_stan[trav_txt]
                
                trav_tree = dct_parse_trees_final[trav_txt]
            
#             trav_tree = Node(parse_trav.label())
#             trav_tree = assign_children(trav_tree, parse_trav[0])
            
                res = simple_distance(seed_tree, trav_tree)
                res_lst.append(res)
            except:
                idx_trav += 1
                continue
            
            idx_trav += 1
        
    dct_results_based_intent[intent] = res_lst
    
file = open('atis/gpt/'+str(NO_TRY)+'/syntax_chaining_res.pkl', 'wb')
# dump information to that file
pickle.dump(dct_results_based_intent, file)
# close the file
file.close()

file = open('atis/gpt/'+str(NO_TRY)+'/syntax_hints.pkl','rb')
(dct_parse_trees_from_stan, dct_parse_trees_final) = pickle.load(file)

df = pd.read_csv('atis/gpt/'+str(NO_TRY)+'/atis_hints.csv').dropna()

dct_intents_with_txts = {}
for index, row in df.iterrows():
    if row['label'] in dct_intents_with_txts:
        dct_intents_with_txts[row['label']].append(row['text'])
    else:
        dct_intents_with_txts[row['label']] = [row['text']]

dct_results_based_intent = {}

for intent in dct_intents_with_txts.keys():
    if intent in dct_results_based_intent:
        continue
    res_lst = []
    lst_seed = dct_intents_with_txts[intent]
    lst_traverse = dct_intents_with_txts[intent]
    print(intent)
    
    for idx, seed_txt in enumerate(lst_seed):
        if idx % 10 == 0:
            print(idx)
            
        try:    
            parse_seed = dct_parse_trees_from_stan[seed_txt]
            
            seed_tree = dct_parse_trees_final[seed_txt]

            idx_trav = idx + 1
        
        except:
            idx_trav = idx + 1
            continue
        
        while (idx_trav < len(lst_traverse)):
            trav_txt = lst_traverse[idx_trav]
            
            # parse_trav = next(parser.raw_parse(trav_txt))
            
            try:
                parse_trav = dct_parse_trees_from_stan[trav_txt]
                
                trav_tree = dct_parse_trees_final[trav_txt]
            
#             trav_tree = Node(parse_trav.label())
#             trav_tree = assign_children(trav_tree, parse_trav[0])
            
                res = simple_distance(seed_tree, trav_tree)
                res_lst.append(res)
            except:
                idx_trav += 1
                continue
            
            idx_trav += 1
        
    dct_results_based_intent[intent] = res_lst
    
file = open('atis/gpt/'+str(NO_TRY)+'/syntax_hints_res.pkl', 'wb')
# dump information to that file
pickle.dump(dct_results_based_intent, file)
# close the file
file.close()
