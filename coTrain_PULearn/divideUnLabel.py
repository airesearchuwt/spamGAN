# -*- coding: utf-8 -*-

import pandas as pd
import random
import numpy as np


print("read unLabeled Data")
label = pd.read_csv("/home/anubha04/dataFiles/NewData/unlabeledClean.tsv",sep='\t',dtype={"Review":str ,  "Product":str})

#30-70
n = int(len(label)*3/10) #divde by 30%
print(n)
all_Ids = np.arange(len(label)) 
random.shuffle(all_Ids)
ten = all_Ids[:n]
ninty = all_Ids[n:]
ten_data = label.loc[ten, :]
ninty_data = label.loc[ninty, :]


print("seventy " + str(len(ninty_data)))

print("save unlabeled70")    
with open("/home/anubha04/dataFiles/NewData/unlabeled70.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(ninty_data.to_csv(sep='\t', index=False))
    
#50-50
n = int(len(label)/2) #divde by 50%
print(n)
all_Ids = np.arange(len(label)) 
random.shuffle(all_Ids)
ten = all_Ids[:n]
ninty = all_Ids[n:]
ten_data = label.loc[ten, :]
ninty_data = label.loc[ninty, :]

print("fifty " + str(len(ninty_data)))

print("save unlabeled50")
with open("/home/anubha04/dataFiles/NewData/unlabeled50.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(ten_data.to_csv(sep='\t', index=False))