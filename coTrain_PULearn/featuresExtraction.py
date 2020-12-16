# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from nltk.tokenize import word_tokenize 
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import sentiwordnet as swn
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('sentiwordnet')

#feature 1
def get_top_ngrams(X_train, y_train, X_test, X_train_u):
    """Select the top 100  unigrams and bigrams from data using Chi squared test.
    """
    print("ngrams")
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    ch2 = SelectKBest(chi2, k=200)
    X_train_new = ch2.fit_transform(X_train, y_train)
    X_test = vectorizer.transform(X_test)
    X_test_new = ch2.transform(X_test)
    X_train_u = vectorizer.transform(X_train_u)
    X_train_u_new =  ch2.transform(X_train_u)
    return X_train_new, X_train_u_new , X_test_new

def ratio_first_secon_person(X):
    """1st vs 2nd Ratio"""
    first_person = ['i', 'my', 'me', 'mine', 'we', 'us', 'our', 'ours']
    second_person = ['you', 'yours', 'your']
    ratio = []
    for x in X:
        tokenlist = word_tokenize(x)
        n_first = len([w for w in tokenlist if w in first_person])
        n_second =len([w for w in  tokenlist if w in second_person])
        if n_second == 0:
            n_second = 0.1
        ratio.append(n_first/n_second)
    ratio = [float(i)/max(ratio) for i in ratio]
    return ratio

def ratio_question_exclamation(X):
    print("?:!")
    question = ['?']
    exclamation = ['!']
    ratio = []
    for x in X:
        tokenlist = word_tokenize(x)
        n_first = len([w for w in tokenlist if w in question])
        n_second =len([w for w in  tokenlist if w in exclamation])
        if n_second == 0:
            n_second = 0.1
        ratio.append(n_first/n_second)
    ratio = [float(i)/max(ratio) for i in ratio]
    return ratio

tagger = PerceptronTagger()
#add feature 6
def compute_sub_posneg(X):
    print("sub & posneg")
    x_subjectivity=[]
    x_posneg=[]
    for sentence in X:
        taggedsentence = []
        obj_score = 0.0
        p_count = 0.0
        n_count = 0.0
        taggedsentence.append(tagger.tag(sentence.split()))
        wnl = nltk.WordNetLemmatizer()
        for idx, words in enumerate(taggedsentence):
            for idx2, t in enumerate(words):
                newtag = ''
                lemmatizedsent = wnl.lemmatize(t[0])
                if t[1].startswith('NN'):
                    newtag = 'n'
                elif t[1].startswith('JJ'):
                    newtag = 'a'
                elif t[1].startswith('V'):
                    newtag = 'v'
                elif t[1].startswith('R'):
                    newtag = 'r'
                else:
                    newtag = ''
                if (newtag != ''):
                    synsets = list(swn.senti_synsets(lemmatizedsent, newtag))
                    score = 0.0
                    obj_wordscore = 0.0
                    if (len(synsets) > 0):
                        for syn in synsets:
                            score += syn.pos_score() - syn.neg_score()
                            obj_wordscore +=syn.obj_score()
                           # print(syn.pos_score, syn.neg_score())
                        score = score / len(synsets)
                        if(score>=0):
                            p_count +=1
                        else:
                            n_count +=1
                        obj_score += obj_wordscore / len(synsets)
                        #print(t, p_count, n_count, obj_score)
        if(n_count==0):
            n_count=1
        x_subjectivity.append(p_count/n_count)
        x_posneg.append(obj_score)
    x_subjectivity = [float(i)/max(x_subjectivity) for i in x_subjectivity]
    x_posneg = [float(i)/max(x_posneg) for i in x_posneg]
    return x_subjectivity, x_posneg