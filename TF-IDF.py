import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import normalize
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

#from scratch ::::::
"""
corpus=[ 'Hey diddle, diddle,',
    'The cow jumped over the moon.',
    'The little dog laughed to see such sport,',
    'and the dish ran away with the spoon. ']

tokenizer=TreeBankWordTokenier()

lexicon=[]

for doc in corpus:
    lexicon+=tokenizer.tokenize(doc.lower())

lexicon=sorted(set([token for token in lexicon if token not in '-.,!?']))

df_tf=pd.DataFrame(
    data=0,
    index=[i for i in range (len(corpus))],
    columns=lexicon
)

for idx, doc in enumerate(corpus):
    tokens=tokenizer.tokenize(doc.lower())
    bag_of_words=Counter(tokens)
    for col in df_tf.columns:
        df_tf.loc[idx,col]=bag_of_words[col]/len(lexicon)


df_idf=pd.DataFrame(
    data=0,
    index=[i for i in range(len(corpus))],
    columns=lexicon
)

num_documents=len(corpus)
for idx,doc in enumerate(corpus):
    for term in lexicon:
        if term in doc.lower():
            df_idf.loc[idx,term]=1

df_idf=np.log((1+num_documents)/(1+np.sum(df_idf,axis=0)))+1

df_tf_idf=df_tf*df_idf
df_tf_idf_norm=pd.DataFrame(
    data=normalize(df_tf_idf.to_numpy(),round(2)),
    columns=lexicon
)
"""
#///////////////////////////////////////////////////////////////

#simple ::::::::

corpus=[ 'Hey diddle, diddle,',
    'The cow jumped over the moon.',
    'The little dog laughed to see such sport,',
    'and the dish ran away with the spoon. ']

vectorizer=TfidfVectorizer()
x= vectorizer.fit_transform(corpus)

df_tf_idf=pd.DataFrame(
    data=x.todense().round(2),
    columns=vectorizer.get_feature_names_out()
)
print(df_tf_idf)

