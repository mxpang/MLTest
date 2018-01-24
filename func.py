import pickle
from collections import Counter
with open('comp_change_proba_sorted.pkl','rb') as handle:
    comp_change_prob=pickle.load(handle)
    
def change_proba_signal(init_comp, comp_tfidf):
    comp_signal = comp_change_prob[init_comp]
    comp_signal = {key:value for key, value in comp_signal.items() if key!=init_comp}
    add_=Counter(comp_signal) + Counter(comp_tfidf)
    add_=sorted(add_, key=add_.get, reverse=True)
    return add_


###extract data from mongo db and save to db
from pymongo import MongoClient
import datetime
import pandas as pd
from datetime import timedelta
from pymongo.errors import BulkWriteError 
client=MongoClient('mongodb://username:password@ip',27017)
col= client['DB']['collection']

pipeline=[{'$unwind':'$logSets'}, {'$match':{'$or':[{'logSets.change_action_desc':'Component entered'}, {'logSets.change_action_desc':'Component changed'}]}}, {'$project':{'_id':1, 'logSets.old_value':1, 'logSets.new_value':1, 'logSets.change_action_desc':1, 'logSets.utc_date':1, 'logSets.utc_time':1}}]
cursor=col.aggregate(pipeline)
#print(pd.DataFrame.from_dict(list(cursor)))
frame=[]

print('start to construct dict...')
for log in cursor:
    single_log={}
    single_log['css_id']=log['_id']
    single_log['date_time']=log['logSets']['utc_date']+log['logSets']['utc_time']
    single_log['change_action_desc']=log['logSets']['change_action_desc']
    single_log['new_value']=log['logSets']['new_value']
    single_log['old_value']=log['logSets']['old_value']
    frame.append(single_log)
print('start to save to dataframe...')
df=pd.DataFrame.from_dict(frame)
print('save to csv...')
df.to_csv('actionLogCom.csv')
print('save done...')

###build markov chain

from collections import Counter, defaultdict

states=df['new_value'].unique()
counts=dict()
for state in states:
    successors = Counter(list(df[df['new_value']==state]['old_value']))
    successors={value:count for value, count in successors.items()}
    counts[state]=successors
for state, successors in counts.items():
    total=sum(successors.values())
    probabilities[state]={s: c/total for s, c in successors.items()}
probabilities