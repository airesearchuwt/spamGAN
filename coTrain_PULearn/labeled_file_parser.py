#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import product_info
import codecs
import data_helper
import pandas as pd

def load_file(hotel_name, filepath):
    review = data_helper.clean_str(filepath[0])
    ratings = product_info.parse_rating(filepath[1])
    num_prod_mentions = []
    mention_count = product_info.count_product_mentions(hotel_name, review)
    num_prod_mentions.append(mention_count)
    return review, num_prod_mentions, ratings


def load_data(main_dir):
    hotel_review_dict = {}
    hotel_rating_dict = {}
    hotel_mention_dict = {}
    hotel_truthful_dict = {}
    review_length_dict = {}
    max_review_length = 0
    data_dict = pd.read_csv(main_dir, sep='\t', dtype={"Review" : str,"Rating" : int, "Label":int,"Product" : str},usecols = [3,4,5,6])
  
    for fle in range(len(data_dict)):
        hotel = product_info.parse_product_name(data_dict.iloc[fle][3])
        if 'ds_store' not in hotel:
            data, prod_mentions, rating = load_file(hotel, data_dict.iloc[fle])
            if hotel not in hotel_review_dict:
                hotel_review_dict[hotel] = []
                hotel_rating_dict[hotel] = []
                hotel_truthful_dict[hotel] = []
                hotel_mention_dict[hotel] = []
                review_length_dict[hotel] = []
                
            review_length = len(data)
            hotel_review_dict[hotel].append(data)
            hotel_rating_dict[hotel].append(rating)
            hotel_truthful_dict[hotel].append(data_dict.iloc[fle][2])
            hotel_mention_dict[hotel].extend(prod_mentions)
            review_length_dict[hotel].append(review_length)
                
            if review_length > max_review_length:
                max_review_length = review_length

    return hotel_review_dict, hotel_rating_dict, hotel_mention_dict, hotel_truthful_dict, review_length_dict, max_review_length