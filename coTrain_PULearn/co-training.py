from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
############################# data processing #####################################################
#change 1 and 0 for spam, non-spam
positive_class_label =1
negative_class_label =0


#features for the 2 classifiers
feature1_cols = [str(i) for i in range(200)] #columns for classifier 1 , if only one classifer replace it with all columns
other_cols = ['firstvssecond','quesexclamation','subjectivity','posneg']
feature1_cols.extend(other_cols)
feature2_cols = ['Ratings','Mentions'] #columns for classifier 2

#Labelled data - Train
data = pd.read_csv('/home/anubha04/dataFiles/100Percent/labeledFeat.tsv', sep='\t')
data['Label'] = data['Label'].apply({-1:1,1:0}.get)

data_test = pd.read_csv('/home/anubha04/dataFiles/100Percent/testFeat.tsv', sep='\t')#nrows = 20)
data_test['Label'] = data_test['Label'].apply({-1:1,1:0}.get)

X_train = data.drop('Label', axis=1).drop('Reviews',axis=1)
y_train = data.Label
X_test = data_test.drop('Label', axis=1).drop('Reviews',axis=1)
y_test = data_test.Label

#classifier 1
x_train_c1 = X_train[feature1_cols]

#classifier 2
x_train_c2 = X_train[feature2_cols]

#unlabelled data - Train
data_train_u = pd.read_csv('/home/anubha04/dataFiles/100Percent/unlabeledFeat.tsv', sep='\t')#nrows = 100)
#print(data_train_u.head())

x_train_c1_u = data_train_u[feature1_cols]
x_train_c2_u = data_train_u[feature2_cols]

#labelled data - Test
x_test_c1 = X_test[feature1_cols]
x_test_c2 = X_test[feature2_cols]

#generates new data of positive and negative reviews from unlabled data
def gen_new_data(model,x_unlabelled,positive,negative):

    positive_index = np.where(model.classes_==positive_class_label)[0][0] #the target class, positive examples
    negative_index = np.where(model.classes_==negative_class_label)[0][0] #negative examples    
    a = model.predict_proba(x_unlabelled)
    p_indices = a[:,positive_index].argsort()[::-1][:positive]
    #gets indices of examples that are classified positive by arranging in decending order of probability and taking p counts
    n_indices = a[:,negative_index].argsort()[::-1][:negative]
    return p_indices,n_indices


#Cotraining-algorithm
#p,n : number of positive and negative examples to be classifies and extracted from unlabeled set
#u : number to unlabeled examples to classify in on run
#k : number of repetations for one training 
def cotraining(p=50,n=50,u=5000,k=500):

    print("in co-training")
    #initial model training with only labeled set
    h1 = GaussianNB().fit(x_train_c1,y_train)
    h2 = GaussianNB().fit(x_train_c2,y_train)

    acc1=[h1.score(x_test_c1,y_test)]
    acc2=[h2.score(x_test_c2,y_test)]
    print("acc1: "+str(acc1)+"\tacc2: "+str(acc2))
    ##MAIN CODE
    from sklearn.utils import resample

    indices = range(x_train_c1_u.shape[0]) 
    #number of unlabelled samples (0,n)
    #print(indices)
    counter=0
    print("in While")
    while counter < k:
        #generate list of indices for u number of samples
        U_prime = resample(indices,n_samples=u)
        
        #getting unlabled data for the generated indices
        temp1 = x_train_c1_u.iloc[U_prime,:]
        temp2 = x_train_c2_u.iloc[U_prime, :]           
        
        #generating new-labelled data (actually indices corresponding to labels) for each classifier
        p_indices1,n_indices1 = gen_new_data(h1,temp1,p,n)
        p_indices2,n_indices2 = gen_new_data(h2,temp2,p,n)

        #extracting the obtained indices from the unlabeled set by combining both the indices predicted by both classifier
        #p_cl1 = temp1.iloc[np.concatenate((p_indices1,p_indices2)), :]
        #n_cl1 = temp1.iloc[np.concatenate((n_indices1,n_indices2)), :]

        #p_cl2 = temp2.iloc[np.concatenate((p_indices1,p_indices2)), :]
        #n_cl2 = temp2.iloc[np.concatenate((n_indices1,n_indices2)), :]

        #removing the incdices from the original unlabled set.
        indices = np.setdiff1d(indices,np.concatenate((p_indices1,p_indices2,n_indices1,n_indices2)))
       
        #fitting new-datapoints
        #h1.partial_fit(pd.concat([p_cl1,n_cl1]),[positive_class_label]*2*p+[negative_class_label]*2*n)
        #h2.partial_fit(pd.concat([p_cl2,n_cl2]),[positive_class_label]*2*p+[negative_class_label]*2*n)

        p_cl1 = temp1.iloc[p_indices1, :]
        n_cl1 = temp1.iloc[n_indices1, :]

        p_cl2 = temp2.iloc[p_indices2, :]
        n_cl2 = temp2.iloc[n_indices2, :]


        #fitting new-datapoints
        h1.partial_fit(pd.concat([p_cl1,n_cl1]),[positive_class_label]*p+[negative_class_label]*n)
        h2.partial_fit(pd.concat([p_cl2,n_cl2]),[positive_class_label]*p+[negative_class_label]*n)

        #Keeping track of accuracy
        acc1.append(h1.score(x_test_c1,y_test))
        acc2.append(h2.score(x_test_c2,y_test))
        y_pred1 = h1.predict_proba(x_test_c1)
        y_pred2 = h2.predict_proba(x_test_c2)
        print("acc1: "+str(acc1[-1])+"\tacc2: "+str(acc2[-1]))
        
        ypred1 = y_pred1[:,0] *y_pred2[:,0]
        ypred2 = y_pred1[:,1] *y_pred2[:,1]

        classlabel = [1 if y2>=y1 else 0 for y1, y2 in zip(ypred1,ypred2)]
        acc = accuracy_score(y_test,classlabel)
        precision, recall, fscore, support = precision_recall_fscore_support(y_test,classlabel)
        f1 = fscore[1]
        counter+=1
    #print("return acc: "+str(acc))
    return acc,f1

if __name__ == '__main__':

    import csv
    import os
    #we run the experiment 10 times
    for j in range(10):
        print("j = "+str(j))
        filename = "/home/anubha04/dataFiles/100Percent/coTrain/traint"+str(j)+".csv"
        accall=[]
        fall=[]
        for i in range(10):
            print("for co training: "+str(i))
            acc,f1 = cotraining()
        
            thisdict =	{
            "iter":i,
            "acc":acc,
            "accstd":0,
            "f1":f1,
            "fstd":0
            }

            accall.append(acc)
            fall.append(f1)
            
            file_exists = os.path.isfile(filename)
            f = open(filename,'a')
            w = csv.DictWriter(f,thisdict.keys())
            if not file_exists:
                print("writing header")
                w.writeheader()
            w.writerow(thisdict)
            f.close()
        meanacc = np.mean(accall)
        stdacc = np.std(fall)
        meanf = np.mean(fall)
        stdf = np.std(fall)
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