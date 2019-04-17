#!/usr/bin/env python
# coding: utf-8
import json
import pandas as pd
import os


def update_data(user_map, json_object, feature_list_toplevel, feature_list_eventproperty, feature_list_userproperty):

    """
    Transform the data from json's encapsulation into a python dict, since there are too many features in amplitude's
    json file, this function also help filter them and only remain those needed

    :param user_map: dict, the dict that records one user's info
    :param json_object: dict, the dict that loads from a json
    :param feature_list_toplevel: list, the features we need in the top level of json_object dict
    :param feature_list_eventproperty: list, the features we need in the event_properties key of json_object
    :param feature_list_userproperty: list, the features we need in the user_properties key of json_object
    """

    for feature in feature_list_toplevel:
        if feature in json_object:
            user_map[feature] = json_object[feature]

    if json_object['event_type'] == 'revenue_amount':
        for feature in feature_list_eventproperty:
            if feature in json_object['event_properties']:
                user_map[feature] = json_object['event_properties'][feature]

    for feature in feature_list_userproperty:
        if feature in json_object['user_properties']:
            user_map[feature] = json_object['user_properties'][feature]

    if json_object['event_type'] == 'Gain Coupon':
        user_map['Gain Coupon'] = True
    elif json_object['event_type'] == 'Usecoupon Click':
        user_map['Use Coupon Click'] = json_object['event_properties']['Source']
    elif json_object['event_type'] == 'Unused Coupon View':
        user_map['Unused Coupon View'] = True

# initialize the feature lists
user_features = ['Account', 'Account Created', 'Default Baby Birthday', 'Baby Count']
user_features_notuserprop = ['city', 'region', 'country', 'device_manufacturer', 'device_type',
                             'apk_channel', 'start_apk_channel', 'version_name']
event_purchase_features = ['UseCoupon', 'CouponPrice', 'CouponType']
event_coupon_features = ['Gain Coupon', 'Use Coupon Click', 'Unused Coupon View']

# load data form json and save in data_map dict
data_map = {}

json_path = '../data/raw/raw_json/'
json_folder = os.listdir(json_path)
for json_file_path in json_folder:
    json_file = open(json_path + json_file_path)
    if json_file_path[-4:] == 'json':
        for json_biobject in json_file:
            json_object = json.loads(json_biobject)
            if json_object['user_id'] in data_map:
                update_data(data_map[json_object['user_id']],
                            json_object,
                            user_features_notuserprop,
                            event_purchase_features,
                            user_features)
            else:
                data_map[json_object['user_id']] = {}
                update_data(data_map[json_object['user_id']],
                            json_object,
                            user_features_notuserprop,
                            event_purchase_features,
                            user_features)

# transform the data from dict to a Pandas DataFrame
df_user_features = pd.DataFrame.from_dict(data_map, orient='index')
print(df_user_features[~df_user_features['UseCoupon'].isnull()])
print(df_user_features.shape)

# save the data to a file
df_user_features.to_csv('user_data_v2.csv')
