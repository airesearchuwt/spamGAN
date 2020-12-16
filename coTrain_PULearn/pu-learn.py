from sklearn.metrics import accuracy_score, precision_recall_fscore_support #classification_report,
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.utils import shuffle

#change 1 and 0 for spam, non-spam
positive_class_label =1
negative_class_label =0

#pu-algorithm
def pu_training(rn):
    
    #Labelled data - Train
    data = pd.read_csv('/home/anubha04/dataFiles/50Percent/labeledFeat.tsv', sep='\t').drop('Reviews',axis=1)
    data['Label'] = data['Label'].apply({-1:1,1:0}.get)

    #Test Data
    data_test = pd.read_csv('/home/anubha04/dataFiles/50Percent/testFeat.tsv', sep='\t').drop('Reviews',axis=1)
    data_test['Label'] = data_test['Label'].apply({-1:1,1:0}.get)

    #unlabelled data - Train
    data_train_u = pd.read_csv('/home/anubha04/dataFiles/50Percent/unlabeledFeat.tsv', sep='\t').drop('Reviews',axis=1)
    
    #shuffling data
    data = shuffle(data)
    data_test = shuffle(data_test)
    data_train_u = shuffle(data_train_u)

    #data for train and test
    data_x = data.drop('Label', axis=1)
    data_y = data.Label

    data_test_x = data_test.drop('Label', axis=1)
    data_test_y = data_test.Label

    #unlabeled set is considered as negative examples.
    y_train_u = pd.DataFrame([negative_class_label]*(data_train_u.shape[0]))

    #extract only the positive examples from labeled set
    indexes = np.where(data_y==positive_class_label)
    
    x_train = data_x.iloc[indexes]
    y_train = data_y.iloc[indexes]

    x_total_train = pd.concat([x_train,data_train_u])
    y_total_train = pd.concat([y_train,y_train_u])
    
    clf =GaussianNB().fit(x_total_train,y_total_train.values.ravel())  
    classlabel = clf.predict(data_test_x)
    acc = accuracy_score(data_test_y,classlabel)
    print("Acc 0: "+str(acc))

    x_train_u_dash = data_train_u
    x_train_u = data_train_u
    print("Size Unlabeled: "+str(len(x_train_u_dash)))

    counter =0    
    print("in while")

    while (x_train_u_dash.shape[0] <= x_train_u.shape[0]) and counter <=10:

        x_train_u = x_train_u_dash
        y_pred = clf.predict(x_train_u_dash)      

        #extracting only negative samples from unlabled set
        indexes = np.where(y_pred == negative_class_label)
        x_train_u_dash = x_train_u_dash.iloc[indexes]
        y_train_u_dash = pd.DataFrame([negative_class_label]*x_train_u_dash.shape[0])
        print("counter: "+str(counter)+"Size Unlabeled: "+str(len(x_train_u_dash)))

        #new training set
        x_total_train= pd.concat([x_train,x_train_u_dash])
        y_total_train = pd.concat([y_train, y_train_u_dash])
       
        clf.fit(x_total_train,y_total_train.values.ravel())

        classlabel = clf.predict(data_test_x)
        
        acc = accuracy_score(data_test_y,classlabel)
        precision, recall, fscore, support = precision_recall_fscore_support(data_test_y,classlabel)
        f1 = fscore[1]
        print("counter: "+str(counter)+"\tAcc: "+str(acc)+"\tF1: "+str(f1))
        counter = counter+1
    return acc, f1


if __name__ == '__main__':
    import csv
    import os
    for j in range(10):
        print("j = "+str(j))
        filename = "/home/anubha04/dataFiles/50Percent/PU_fit_try/putrainnb"+str(j)+".csv"
        accall1=[]
        fall1=[]
        for i in range(50):
            acc,f1 = pu_training(i)
            thisdict =	{
            "iter":i,
            "acc":acc,
            "accstd":0,
            "f1":f1,
            "fstd":0
            }

            accall1.append(acc)
            fall1.append(f1)
            file_exists = os.path.isfile(filename)
            f = open(filename,'a')
            w = csv.DictWriter(f,thisdict.keys())
            if not file_exists:
                print("writing header")
                w.writeheader()
            w.writerow(thisdict)
            f.close()
        meanacc = np.mean(accall1)
        stdacc = np.std(accall1)
        meanf = np.mean(fall1)
        stdf = np.std(fall1)
        print(meanacc, stdacc, meanf,stdf)
        thisdict =	{
            "iter":100,
            "acc":meanacc,
            "accstd":stdacc,
            "f1":meanf,
            "fstd":stdf
            }
        file_exists = os.path.isfile(filename)
        f = open(filename,'a')
        w = csv.DictWriter(f,thisdict.keys())
        if not file_exists:
            print("writing header")
            w.writeheader()
        w.writerow(thisdict)
        f.close()