#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs
import re
import os 
import product_info 
import data_helper
import pandas as pd


def load_file(filename):
    hotel_name = product_info.parse_product_name(re.sub(r"_", " ", filename[2]).lower())
    ratings = product_info.parse_rating(filename[1])
    hotel_data = data_helper.clean_str(filename[0])
    num_prod_mentions = []
    mention_count = product_info.count_product_mentions(hotel_name, hotel_data)
    num_prod_mentions.append(mention_count)
    return hotel_name, hotel_data, ratings, num_prod_mentions


def load_data(directory_path):
    hotel_review_dict = {}
    hotel_rating_dict = {}
    hotel_mention_dict = {}
    review_length_dict = {}
    max_review_length = 0
    data_dict = pd.read_csv(directory_path, sep='\t', dtype={"Review" : str,"Rating" : int, "Product" : str},usecols = [3,4,6])
    for fle in range(len(data_dict)):
        hotel, data, ratings, mentions = load_file(data_dict.iloc[fle])
        review_length = len(data)
        if hotel not in hotel_review_dict:
            hotel_review_dict[hotel] = []
            hotel_rating_dict[hotel] = []
            hotel_mention_dict[hotel] = []
            review_length_dict[hotel] = []
        hotel_review_dict[hotel].append(data)
        hotel_rating_dict[hotel].append(ratings)
        hotel_mention_dict[hotel].extend(mentions)
        review_length_dict[hotel].append(len(data))
        if review_length > max_review_length:
            max_review_length = review_length
    return hotel_review_dict, hotel_rating_dict, hotel_mention_dict, review_length_dict, max_review_length