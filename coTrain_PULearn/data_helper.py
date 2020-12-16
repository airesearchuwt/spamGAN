#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np

def clean_str(raw_data):
    string = raw_data.replace('\n','')
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_X_y(reviews, ratings, mentions, lengths, truthful):
    X_reviews = []
    X_ratings = []
    X_mentions = []
    X_lengths = []
    y = []
    for k in reviews.keys():
        X_reviews.extend(reviews[k])
        X_ratings.extend(ratings[k])
        X_mentions.extend(mentions[k])
        X_lengths.extend(lengths[k])
        if truthful is not None:
            y.extend(truthful[k])
    #print(np.shape(X_reviews))
    
    return X_reviews, X_ratings, X_mentions, X_lengths, y

def normalize_review_lengths(unlabeled_max, labeled_max, test_max, unlabeled_lengths, labeled_lengths,test_lengths):
    max_review_length = float(max(unlabeled_max, labeled_max, test_max))
    for k, v in unlabeled_lengths.items():
        v[:] = [l / max_review_length for l in v]
    for k, v in labeled_lengths.items():
        v[:] = [l / max_review_length for l in v]
    for k, v in test_lengths.items():
        v[:] = [l / max_review_length for l in v]
    return unlabeled_lengths, labeled_lengths, test_lengths

def create_product_arrays(ratings, mentions):
    data = [ratings, mentions]
    data = np.array(data)
    print(data)
    print(data.shape)
    return data