#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#this file is used to extarct features from the reviews for co-training

import unlabeled_file_parser
import labeled_file_parser
import data_helper
import product_info
import featuresExtraction
import pandas as pd

#loactions to files containing data
LABELED_DIR = "/home/anubha04/dataFiles/NewData/labeled10.tsv"
UNLABELED_PATH = "/home/anubha04/dataFiles/NewData/unlabeled50.tsv"
TEST_DIR = "/home/anubha04/dataFiles/NewData/labeledTest.tsv"

data_dict_la = pd.DataFrame()
data_dict_un = pd.DataFrame()
data_dict_test = pd.DataFrame()

print("load Data unlabeled")
unlabeled_review_dict, unlabeled_rating_dict, unlabeled_mention_dict, unlabeled_lengths, unlabeled_max_len = unlabeled_file_parser.load_data(UNLABELED_PATH)

#print(len(unlabeled_review_dict))

print("load Data labeled")
labeled_review_dict, labeled_rating_dict, labeled_mention_dict, labeled_truthful_dict, labeled_lengths, labeled_max_len = labeled_file_parser.load_data(LABELED_DIR)


print("load Data test")
test_review_dict, test_rating_dict, test_mention_dict, test_truthful_dict, test_lengths, test_max_len = labeled_file_parser.load_data(TEST_DIR)

#print("normalize review length")
unlabeled_lengths, labeled_lengths, test_lengths = data_helper.normalize_review_lengths(unlabeled_max_len, labeled_max_len, test_max_len, unlabeled_lengths, labeled_lengths, test_lengths)

#print("normalize mention count")
unlabeled_mention_dict, labeled_mention_dict, test_mention_dict = product_info.normalize_mention_counts(unlabeled_mention_dict, labeled_mention_dict, test_mention_dict)

X_labeled_reviews, X_labeled_ratings, X_labeled_mentions, X_labeled_lengths, y_labeled = data_helper.get_X_y(labeled_review_dict, labeled_rating_dict, labeled_mention_dict, labeled_lengths, labeled_truthful_dict)

data_dict_la['Reviews'] =  pd.Series(X_labeled_reviews)
data_dict_la['Ratings'] =  pd.Series(X_labeled_ratings)
data_dict_la['Mentions'] =  pd.Series(X_labeled_mentions)
data_dict_la['Length'] =  pd.Series(X_labeled_lengths)
data_dict_la['Label'] =  pd.Series(y_labeled)

X_test_reviews, X_test_ratings, X_test_mentions, X_test_lengths, y_test = data_helper.get_X_y(test_review_dict, test_rating_dict, test_mention_dict, test_lengths, test_truthful_dict)

data_dict_test['Reviews'] =  pd.Series(X_test_reviews)
data_dict_test['Ratings'] =  pd.Series(X_test_ratings)
data_dict_test['Mentions'] =  pd.Series(X_test_mentions)
data_dict_test['Length'] =  pd.Series(X_test_lengths)
data_dict_test['Label'] =  pd.Series(y_test)

X_unlabeled_reviews, X_unlabeled_ratings, X_unlabeled_mentions, X_unlabeled_lengths, placeholder = data_helper.get_X_y(unlabeled_review_dict, unlabeled_rating_dict, unlabeled_mention_dict, unlabeled_lengths, None)

data_dict_un['Reviews'] =  pd.Series(X_unlabeled_reviews)
data_dict_un['Ratings'] =  pd.Series(X_unlabeled_ratings)
data_dict_un['Mentions'] =  pd.Series(X_unlabeled_mentions)
data_dict_un['Length'] =  pd.Series(X_unlabeled_lengths)
#data_dict_un['Label'] =  pd.Series(y_labeled)

print("test train data")
X_review_train = X_labeled_reviews
X_review_test = X_test_reviews
y_train = y_labeled
y_test = y_test

X_labeled_lengths1 = []
for i in X_labeled_lengths:
    X_labeled_lengths1.append([i])

X_labeled_lengths2 = []
for i in X_test_lengths:
    X_labeled_lengths2.append([i])

X_labeled_ratings1 = X_labeled_ratings
X_labeled_ratings2 = X_test_ratings
X_labeled_mentions1 = X_labeled_mentions
X_labeled_mentions2 = X_test_mentions


#print("feature1 : ngrams")
X_review_train_ngr, X_review_train_u_ngr , X_review_test_ngr = featuresExtraction.get_top_ngrams(X_review_train, y_train, X_review_test, X_unlabeled_reviews) #feature 1

data_dict_la = pd.concat([data_dict_la, pd.DataFrame(X_review_train_ngr.toarray())], axis=1)

data_dict_un = pd.concat([data_dict_un, pd.DataFrame(X_review_train_u_ngr.toarray())], axis=1)

data_dict_test = pd.concat([data_dict_test, pd.DataFrame(X_review_test_ngr.toarray())], axis=1)


#print("feature3 : 1st vs 2nd")
x_firstvssecond_train = featuresExtraction.ratio_first_secon_person(X_review_train)
data_dict_la['firstvssecond'] =  pd.Series(x_firstvssecond_train)

x_firstvssecond_train_u = featuresExtraction.ratio_first_secon_person(X_unlabeled_reviews)
data_dict_un['firstvssecond'] =  pd.Series(x_firstvssecond_train_u)

x_firstvssecond_test = featuresExtraction.ratio_first_secon_person(X_review_test)
data_dict_test['firstvssecond'] =  pd.Series(x_firstvssecond_test)


#print("feature 5 : ?:!")
x_quesexclamation_train = featuresExtraction.ratio_question_exclamation(X_review_train)
data_dict_la['quesexclamation'] =  pd.Series(x_quesexclamation_train)

x_quesexclamation_train_u = featuresExtraction.ratio_question_exclamation(X_unlabeled_reviews)
data_dict_un['quesexclamation'] =  pd.Series(x_quesexclamation_train_u)

x_quesexclamation_test = featuresExtraction.ratio_question_exclamation(X_review_test)
data_dict_test['quesexclamation'] =  pd.Series(x_quesexclamation_test)

#print("feature 6, 7 : sub and posneg")
x_subjectivity_train, x_posneg_train = featuresExtraction.compute_sub_posneg(X_review_train)
data_dict_la['subjectivity'] =  pd.Series(x_subjectivity_train)
data_dict_la['posneg'] =  pd.Series(x_posneg_train)

x_subjectivity_train_u, x_posneg_train_u = featuresExtraction.compute_sub_posneg(X_unlabeled_reviews)
data_dict_un['subjectivity'] =  pd.Series(x_subjectivity_train_u)
data_dict_un['posneg'] =  pd.Series(x_posneg_train_u)

x_subjectivity_test, x_posneg_test = featuresExtraction.compute_sub_posneg(X_review_test)
data_dict_test['subjectivity'] =  pd.Series(x_subjectivity_test)
data_dict_test['posneg'] =  pd.Series(x_posneg_test)

#print("save labeled")
with open("/home/anubha04/dataFiles/10Percent/labeledFeat.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(data_dict_la.to_csv(sep='\t', index=False))
    
print("save unlabeled")
with open("/home/anubha04/dataFiles/10Percent/unlabeledFeat.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(data_dict_un.to_csv(sep='\t', index=False))
    
#print("save test")
with open("/home/anubha04/dataFiles/10Percent/testFeat.tsv",'w',encoding="utf-8") as write_csv:
    write_csv.write(data_dict_test.to_csv(sep='\t', index=False))
print("KO!")