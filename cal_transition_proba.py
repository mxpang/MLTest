import pandas as pd
import pickle

df2=pd.read_csv('file_name.csv')

def dump_file(dicts, file_name):
    with open(file_name,'wb') as handle:
        pickle.dump(dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
#fill na with new value
def fill_na(row):
    if(pd.isna(row[4])):
        row[4]=row[3]
    return row

df2 = df2.apply(lambda x:fill_na(x), axis=1)

#build transition frequencies dictionary
num_dict = {}
l1 = list(df2['new_value'].apply(lambda x:x.lower() if pd.notna(x) else x))
l2 = list(df2['old_value'].apply(lambda x:x.lower() if pd.notna(x) else x))
com_list = zip(l2,l1)
for t in com_list:
    if t[0] in num_dict:
        if t[1] in num_dict[t[0]]:
            num_dict[t[0]][t[1]] = num_dict[t[0]][t[1]]+1
        else:
            num_dict[t[0]][t[1]] = 1
    else:
        num_dict[t[0]] = {}
        num_dict[t[0]][t[1]] = 1

#calculate transition probabilities
proba = {}
for state, successors in num_dict.items():
    total = sum(successors.values())
    proba[state] = {s: c/total for s, c in successors.items()}
        
#build trainsition probabilities using conditional probabilities
conditional_prob = {}
for key, value in proba.items():
    total = sum(v for k, v in value.items() if k!=key)
    conditional_prob[key] = {k:(v/total) for k, v in value.items() if k!=key}
    biggest = sorted(conditional_prob[key], key=conditional_prob[key].get, reverse=True)
    if len(biggest):
        conditional_prob[key][key] = conditional_prob[key][biggest[0]]
    else:
        continue

#build trainsition probabilities using cube root approach        
cube_root_proba = {}
for key, value in proba.items():
    total = sum(v**(1/3) for v in value.values())
    cube_root_proba[key] = {k:(v**(1/3)/total) for k, v in value.items()}
    
dump_file(conditional_prob, 'conditional_proba.pkl')
dump_file(cube_root_proba, 'cube_root_proba.pkl')
