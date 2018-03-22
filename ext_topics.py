import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from collections import Counter
import jieba
jieba.load_userdict('added_voca.txt')
#, header=None, names=['task_id','content','task_amount','label','scene','swindle_way'])
df=pd.read_csv()

df.drop(df[(df['uid'].isnull()) & (df['oid'].isnull())].index, axis=0, inplace=True)
df.drop(df[df['field']=='nan'].index, axis=0, inplace=True)
df.drop(df[df['field']==np.nan].index, axis=0, inplace=True)
df.drop(df[df['field'].isnull()].index, axis=0, inplace=True)
df.drop(['uid','oid'], axis=1, inplace=True)

#clean empty values
ids=[]
def clean_null(row,task_ids):
    if not row['field']:
        ids.append(row['field'])
df.apply(clean_null, axis=1, args=(ids,))
for taskid in taskids:
    df.drop(df[df['task_id']==taskid].index, axis=0, inplace=True)

#only keep chinese
def get_ch(row):
    row=row.strip()#.decode('utf-8', 'ignore')
    patt=re.compile(r'[^\u4e00-\u9fa5]')
    row=' '.join([w for w in patt.split(row) if len(w)>1]).strip()
    return row
df['field_zh']=df['field'].apply(get_ch)


#concate string by columns
df['field'] = df['field'].fillna('') + df['field'].fillna('')
df.drop(['field'], axis=1, inplace=True)
def split_(row):
    return row.split('-')[0]
df['field']=df.field.apply(split_)


#split words
sw = pd.read_csv('sw.txt', header=None)
sw = sw.values[:,0]

## split
def split(row):
    row=re.sub('[,，。:.]',' ',str(row))
    w_l=jieba.cut(str(row),cut_all=False)
    row=[w for w in w_l if w not in sw]
    return ' '.join(w for w in row)
df['field']=df['field'].apply(split)

##build vec
vec=TfidfVectorizer(max_df=0.95, min_df=2, decode_error='ignore', norm='l2', sublinear_tf=True, use_idf=True)
vec.fit(df['field_zh'])
features=vec.get_feature_names()

def get_tfidf_score(vec, dfToTrans, perc): #series
    tfidf_score=vec.transform(dfToTrans).toarray()#.values[:,0]
    tfidf_score_sum= np.sum(tfidf_score, axis=0)
    #use the max tfidf score
    #tfidf_score_sum= np.max(tfidf_score, axis=0)
    word_count = np.count_nonzero(tfidf_score, axis=0)
    words_freq=dict(zip(features, word_count))
    feat = dict(zip(features, tfidf_score_sum))
    #feats = {feat_name: tfidf_score_sum[i] for i,feat_name in enumerate(features)}
    return feat, words_freq
    
def get_feats(df, vec):  ###feats=[feat1, feat2...] feats[0]-fraud
    feats=[]
    feats_count=[]
    for risk_meth in df['types'].unique():
        temp=df[df['types']==risk_meth]['field_zh'].values
        feat, feat_count=get_tfidf_score(vec, temp, 0)
        feats.append(feat)
        feats_count.append(feat_count)
    return feats, feats_count

def get_common_words(df, vec):
    words_l=[]
    feats,nums = get_feats(df, vec)
    for i,feat in enumerate(feats):
        words_l.append([key for key in feat if nums[i][key]>0])
    #get dunplicated words
    common_words=[]
    for i,words in enumerate(words_l):
        if i < len(words_l)-1:
            common_words.extend(set(words_l[i]).intersection(words_l[i+1]))
    return common_words

common_words = get_common_words(df, vec)

def clean_common_words(featsn, common_words):
    new={k:v for k,v in featsn.items() if k not in common_words}
    #sorted_new=sorted(new.items(), key=lambda x:x[1], reverse=True)
    return new
top_feats = clean_common_words(feats[0],common_words)
final={}
for k, v in top_feats.items():
    if words_freq[0][k]>0.001:
        final[k]=v
