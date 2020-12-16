#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

def parse_product_name(prod_name):
    prod_name = prod_name.lower()
    prod_name_words = re.findall("\w+", prod_name) #converts string to words array
    prod_name_words = [word for word in prod_name_words if word not in stopwords.words('english')] #removes english stopwords
    return ' '.join(prod_name_words)

def count_product_mentions(prod_name, review_text):
    """Counts the number of times a product name is mentioned in a review by taking bigram counts.
    """
    mention_count = 0
    hotel_words = prod_name.split()
    for w in hotel_words:
        mention_count += review_text.count(w)
    return mention_count

def parse_rating(rating):
    """Determines if rating was positive or not. 
    Returns 1 for 3, 4, or 5 stars, 0 otherwise
    """
    if int(rating) > 2:
        return 1
    else:
        return 0

def normalize_mention_counts(unlabeled_mentions, labeled_mentions, test_mentions):
    """Normalizes each count by maximum out of all counts.
    """
    overall_max = 0
    for k,v in unlabeled_mentions.items():
        hotel_max = max(v, default = 0)
        if hotel_max > overall_max:
            overall_max = hotel_max
            
    for k,v in labeled_mentions.items():
        hotel_max = max(v)
        if hotel_max > overall_max:
            overall_max = hotel_max
            
    for k,v in test_mentions.items():
        hotel_max = max(v)
        if hotel_max > overall_max:
            overall_max = hotel_max
            
    for k,v in unlabeled_mentions.items():
        v[:] = [m/float(overall_max) for m in v]
        
    for k,v in labeled_mentions.items():
        v[:] = [m/float(overall_max) for m in v]
        
    for k,v in test_mentions.items():
        v[:] = [m/float(overall_max) for m in v]
        
    return unlabeled_mentions, labeled_mentions, test_mentions