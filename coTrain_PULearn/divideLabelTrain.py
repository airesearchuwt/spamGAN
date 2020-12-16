# -*- coding: utf-8 -*-

import pandas as pd
import random
import numpy as np

print("read Labeled Data")
label = pd.read_csv("/home/anubha04/dataFiles/NewData/labeledTrain.tsv",sep='\t')


#10-90

n = int(len(label)*1/10) #divde by 10%
print(n)
all_Ids = np.arange(len(label)) 
random.shuffle(all_Ids)
ten = all_Ids[:n]
ninty = all_Ids[n:]
ten_data = label.loc[ten, :]
ninty_data = label.loc[ninty, :]

print("ten " + str(len(ten_data)))
print("ninty " + str(len(ninty_data)))

print("save labeled10")
with open("/home/anubha04/dataFiles/NewData/labeled10.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(ten_data.to_csv(sep='\t', index=False))

print("save labeled90")    
with open("/home/anubha04/dataFiles/NewData/labeled90.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(ninty_data.to_csv(sep='\t', index=False))

#30-70
n = int(len(label)*3/10) #divde by 30%
print(n)
all_Ids = np.arange(len(label)) 
random.shuffle(all_Ids)
ten = all_Ids[:n]
ninty = all_Ids[n:]
ten_data = label.loc[ten, :]
ninty_data = label.loc[ninty, :]

print("ten " + str(len(ten_data)))
print("ninty " + str(len(ninty_data)))

print("save labeled30")
with open("/home/anubha04/dataFiles/NewData/labeled30.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(ten_data.to_csv(sep='\t', index=False))

print("save labeled70")    
with open("/home/anubha04/dataFiles/NewData/labeled70.tsv",'w',encoding="utf-8") as write_csv:
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

print("ten " + str(len(ten_data)))
print("ninty " + str(len(ninty_data)))

print("save labeled50")
with open("/home/anubha04/dataFiles/NewData/labeled50.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(ten_data.to_csv(sep='\t', index=False))
