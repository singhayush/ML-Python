# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 19:18:32 2017

@author: AY370389
"""
from nltk.chunk import ne_chunk
from nltk import sent_tokenize, word_tokenize, pos_tag
import pandas as pd
#import numpy as np

#sentence and word tokenization

sentence=""
df=pd.read_csv('data_in.csv')

for i in range(df.shape[0]):
    sentence+=df["Comment"][i].strip('"')+" "
result=list(sent_tokenize(sentence))

result1=list(word_tokenize(sentence))

df1=pd.DataFrame({'Sentences':result})
df2=pd.DataFrame({'Words':result1})
final=pd.concat([df1,df2],axis=1)

final.to_csv("data_out.csv")

#NE chunking, POS tagging and displaying tree structure

pstag=pos_tag(word_tokenize(sentence))
chunksen = ne_chunk(pstag)
print (chunksen)
chunksen.draw()
print(pstag)

#stemming and lemmatization
df3=pd.read_csv("data.txt",header=None,delimiter='\n')
print(df3)
#removing stop words
from nltk.corpus import stopwords
from nltk import stem
sen=""
for i in range(df3.shape[0]):
    sen+=df3[0][i]+" "

stop = set(stopwords.words('english'))
nonstop=[i for i in word_tokenize(sen.lower()) if i not in stop]

stemm=[]
ps = stem.SnowballStemmer('english')
for word in nonstop:
    stemm.append(ps.stem(word))
    
lemm=[]
wnlem=stem.WordNetLemmatizer()
for word in nonstop:
    lemm.append(wnlem.lemmatize(word))
    
#comparing result
z=zip(stemm,lemm)
print(set(z))

#sentiment analyis

senti_dict = {}
for each_line in open('sentidict.txt'):
    word,score = each_line.split('\t')
    senti_dict[word] = int(score)

for line in open('analysis5.txt'):
    words="".join(line.lower().split('.')).split()
    print(words)
    score=sum(senti_dict.get(word, 0) for word in words)
    print('positive' if score>0 else 'negative' , score)

#