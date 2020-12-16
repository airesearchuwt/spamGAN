# -*- coding: utf-8 -*-

import pandas as pd
import random
import numpy as np

print("read Labeled Data")
label = pd.read_csv("/home/anubha04/dataFiles/NewData/labeledClean.tsv", sep='\t')
print(len(label))
n = int(len(label)*8/10) #divde by 80%
print(n)
all_Ids = np.arange(len(label)) 
random.shuffle(all_Ids)
train = all_Ids[:n]
test = all_Ids[n:]
train_data = label.loc[train, :]
test_data = label.loc[test, :]

print("train " + str(len(train_data)))
print("test " + str(len(test_data)))

print("save test")
with open("/home/anubha04/dataFiles/NewData/labeledTest.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(test_data.to_csv(sep='\t', index=False))

print("save train")    
with open("/home/anubha04/dataFiles/NewData/labeledTrain.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(train_data.to_csv(sep='\t', index=False))