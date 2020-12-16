# -*- coding: utf-8 -*-
import pandas as pd
import random
import numpy as np
import re

#funtion to clean the strings in the input data. Namely Reviews and Product Names
def clean(s,):
    ns = s.lower()
    ns = re.sub('[0-9]+', 'N', ns)
    ns = re.sub('[^a-zA-Z0-9 \-.,\'\"!?()]', ' ', ns) # Eliminate all but these chars
    ns = re.sub('([.,!?()\"\'])', r' \1 ', ns) # Space out punctuation
    ns = re.sub('\s{2,}', ' ', ns) # Trim ws
    str.strip(ns)
    return ns

print("read Labeled Data")
label = pd.read_csv("/home/anubha04/dataFiles/NewData/labeled.csv",sep=',')

print("read unLabeled Data")
unlabel = pd.read_csv("/home/anubha04/dataFiles/NewData/unlabeled.csv",sep=',')

print("Clean Data")

for i in range(len(label)):
    label.loc[i,'Review'] = clean(label.loc[i,'Review'])
    label.loc[i,'Product'] = clean(label.loc[i,'Product'])
    print("label " +str(i))
    
for i in range(len(unlabel)):
    unlabel.loc[i,'Review'] = clean(unlabel.loc[i,'Review'])
    unlabel.loc[i,'Product'] = clean(unlabel.loc[i,'Product'])
    print("unlabel "+ str(i))
   
    
print("save labeled")
with open("/home/anubha04/dataFiles/NewData/labeledClean.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(label.to_csv(sep='\t', index=False))

print("save unlabeled")    
with open("/home/anubha04/dataFiles/NewData/unlabeledClean.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(unlabel.to_csv(sep='\t', index=False))
