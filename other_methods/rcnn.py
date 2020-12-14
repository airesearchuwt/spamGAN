from subprocess import check_output
from nltk.tokenize import sent_tokenize, word_tokenize
# Any results you write to the current directory are saved as output.

import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve, precision_recall_fscore_support
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf


#set parameters
embed_size = 50 # how big is each word vector
max_features = 25000 # how many unique words to use (i.e num rows in embedding vector)
train_review_path = "/home/yankun/spamGAN/dividedData/labeled50/train_review.txt"
train_label_path = "/home/yankun/spamGAN/dividedData/labeled50/train_label.txt"
val_review_path = "/home/yankun/spamGAN/dividedData/labeled50/val_review.txt"
val_label_path = "/home/yankun/spamGAN/dividedData/labeled50/val_label.txt"
test_review_path = "/home/yankun/spamGAN/dividedData/test_review.txt"
test_label_path = "/home/yankun/spamGAN/dividedData/test_label.txt"
embedding_file_path = "/home/yankun/rnn/embedding/glove.6B.50d.txt"
result_path = "/home/yankun/rnn/result/result.csv"

for i in range(8):
    def load_dataset():
        train_review, train_label, val_review, val_label, test_review, test_label= [],[],[],[],[],[]
        with open(train_review_path, 'r') as f1, open(train_label_path, 'r') as f2:
            txts = f1.readlines()
            labs = f2.readlines()
            train_review = [txt for txt in txts]
            train_label = [int(lab) for lab in labs]
        with open(val_review_path, 'r') as f1 ,open(val_label_path, 'r') as f2:
            txts = f1.readlines()
            labs = f2.readlines()
            val_review = [txt for txt in txts]
            val_label = [int(lab) for lab in labs]
        with open(test_review_path, 'r') as f1 ,open(test_label_path, 'r') as f2:
            txts = f1.readlines()
            labs = f2.readlines()
            test_review = [txt for txt in txts]
            test_label = [int(lab) for lab in labs]
            return train_review, train_label, val_review, val_label, test_review, test_label

    train_review, train_label, val_review, val_label, test_review, test_label = load_dataset()
    arr=[1 if i == 0 else 0 for i in train_label]
    y_train = np.stack((train_label, arr), axis=1)

    arr2 = [1 if i == 0 else 0 for i in test_label]
    y_test = np.stack((test_label, arr2), axis=1)

    arr3 = [1 if i == 0 else 0 for i in val_label]
    y_val = np.stack((val_label, arr3), axis=1)


    #get the max length of sentences
    sentences = train_review
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    maxlength = max(len(sentences) for sentences in tokenized_sentences)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_review))
    list_tokenized_train = tokenizer.texts_to_sequences(train_review)
    list_tokenized_val = tokenizer.texts_to_sequences(val_review)
    list_tokenized_test = tokenizer.texts_to_sequences(test_review)
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlength)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlength)
    X_val = pad_sequences(list_tokenized_val, maxlen=maxlength)

    val = (np.array(X_val), y_val)

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file_path, 'r'))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    #assign embedding vector to each word
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    print(nb_words, len(word_index), embedding_matrix.shape)
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i-1] = embedding_vector

    max_features = min(max_features, len(word_index))
    checkpoint = ModelCheckpoint('/home/yankun/rnn/output/{acc:.4f}.hdf5', monitor='acc', verbose=1, save_best_only=True, mode='auto')
    inp = Input(shape=(maxlength,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(2, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_t, y_train, batch_size=64, epochs=7, callbacks=[checkpoint,EarlyStopping(monitor='acc', patience=3,restore_best_weights=True)],validation_split=0.1, validation_data=val)

    #best = save_best_model(10, "output", 5, ".hdf5")

    y_pred = model.predict([X_te], batch_size=32, verbose=1)


    #print(y_pred.shape)

    y_pred_keras = y_pred[:,0]
    #print(y_pred_keras)
    # print(y_test)
    # print(y_test[:,0])
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test[:,0], y_pred_keras, pos_label=0)

    class_label = [ 1 if y[0]>=y[1]  else 0 for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test[:,0], class_label, labels=[0,1]).ravel()
    print(confusion_matrix(y_test[:,0], class_label, labels=[0,1]))
    print(auc(fpr_keras,tpr_keras))
    precision, recall, f_score, support = precision_recall_fscore_support(y_test[:,0],class_label)
    print(precision, recall, f_score)
    #print(y_pred, y_pred.argmax(axis=1))
    print(accuracy_score(y_test[:,0],class_label))


    thisdict =	{
        "tn": tn,
        "tp":tp,
        "fp":fp,
        "fn":fn,
        "auc": auc(fpr_keras,tpr_keras),
        "precision +ve": precision[0],
        "recall +ve": recall[0],
        "fscore +ve": f_score[0],
        "accuracy": accuracy_score(y_test[:,0],class_label),
        "precision -ve": precision[1],
        "recall -ve": recall[1],
        "fscore -ve": f_score[1],
    }

    import csv
    file_exists = os.path.isfile(result_path)
    f = open(result_path,'a')
    w = csv.DictWriter(f, thisdict.keys())
    if not file_exists:
        print("writing header")
        w.writeheader()
    w.writerow(thisdict)
    f.close()
