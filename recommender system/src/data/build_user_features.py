# This script is used to clean data, do feature engineering and use them to get a logistic regression model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import feature_selection
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from datetime import datetime


# encoding methods:
def encode_region(df_model_input, df_user):

    """
    Encode the geographical features,temporarily only the region column.

    :param df_user: Pandas DataFrame, the user properties
    :param df_model_input: Pandas DataFrame, the DataFrame that fit the format of model's input data
    """

    # initialize the dicts used in this function, they record the map between initial format and the model input format
    notchinamainland_map = {}
    region_encode_map = {'Guangdong': 1, 'Zhejiang': 2, 'Jiangsu': 3, 'Shanghai': 4, 'Beijing': 10, 'Southwest': 5,
                         'Northwest': 6, 'Northeast': 7, 'Northchina': 8, 'Middlechina': 9, 'notchinamainland': 0}
    province_region_map = {'Guangdong': 'Guangdong', 'Zhejiang': 'Zhejiang', 'Jiangsu': 'Jiangsu',
                           'Shanghai': 'Shanghai', 'Sichuan': 'Southwest', 'Chongqing': 'Southwest',
                           'Guizhou': 'Southwest', 'Yunnan': 'Southwest', 'Guangxi': 'Southwest',
                           'Inner Mongolia Autonomous Region': 'Northeast', 'Liaoning': 'Northeast',
                           'Jilin': 'Northeast', 'Heilongjiang': 'Northeast', 'Xinjiang': 'Northwest',
                           'Gansu': 'Northwest', 'Qinghai': 'Northwest', 'Shaanxi': 'Northeast',
                           'Ningxia Hui Autonomous Region': 'Northwest', 'Shanxi': 'Northchina',
                           'Shandong': 'Northchina', 'Hebei': 'Northchina', 'Tianjin': 'Northchina',
                           'Henan': 'Northchina', 'Hubei': 'Middlechina', 'Hunan': 'Middlechina',
                           'Anhui': 'Middlechina', 'Jiangxi': 'Middlechina', 'Fujian': 'Middlechina',
                           'Hainan': 'Middlechina', 'Beijing': 'Beijing'}

    for region in df_user['region'].unique():
        if region not in province_region_map:
            notchinamainland_map[region] = 'notchinamainland'
    df_model_input['region'] = df_user['region'].map(province_region_map).map(region_encode_map)


def encode_device(df_model_input, df_user):

    """
    Encode the device info, which includes iPad, iPhone, Android Phone and others(may be added in the future).

    :param df_user: Pandas DataFrame, the user properties
    :param df_model_input: Pandas DataFrame, the DataFrame that fit the format of model's input data
    """

    device_encode_map = {'ipad': 0, 'iphone': 1, 'others': 2}
    device_type_map = {}
    for device_type in df_user['device_type'].unique():
        if 'iPad' in device_type:
            device_type_map[device_type] = 'ipad'
        elif 'iPhone' in device_type:
            device_type_map[device_type] = 'iphone'
        else:
            device_type_map[device_type] = 'others'
    df_model_input['device_type'] = df_user['device_type'].map(device_type_map).map(device_encode_map)


def version_encode(df_model_input, df_user):

    df_model_input['version_name'] = df_user['version_name']
    df_model_input['version_name'] = df_model_input['version_name'].apply(lambda x: int(x[0]))


def account_encode(df_model_input, df_user):

    account_encode_map = dict(zip(df_user['Account'].unique(),
                                  [i for i in range(len(df_user['Account'].unique()))]))
    df_model_input['Account'] = df_user['Account'].map(account_encode_map)


def time_delta_timestamp(now, user_time):
    return (now - datetime.strptime(user_time, "%Y-%m-%d %H:%M:%S +%f")).days


def time_transform_abs2rela(df_model_input, df_user):
    current_time = datetime.now()
    df_model_input['Account Created'] = df_user['Account Created'].\
        apply(lambda x: time_delta_timestamp(current_time, x)//365+1)


def encode_baby_birtday(df_model_input, df_user):
    now = datetime.now()
    df_model_input['Default Baby Birthday'] = df_user['Default Baby Birthday'].\
        astype('int').astype('str').apply(lambda x: x[0:4]+'-'+x[4:])
    df_model_input['Default Baby Birthday'] = df_model_input['Default Baby Birthday'].\
        apply(lambda x: (now - datetime.strptime(x, "%Y-%m")).days//365+1)


def encode_baby_count(df_model_input, df_user):
    df_model_input['Baby Count'] = df_user['Baby Count']
    df_model_input['Baby Count'].loc[df_model_input['Baby Count'] == 1] = 1
    df_model_input['Baby Count'].loc[df_model_input['Baby Count'] == 2] = 2
    df_model_input['Baby Count'].loc[(df_model_input['Baby Count'] != 1) & (df_model_input['Baby Count'] != 2)] = 0


def encode_coupon_price(df_model_input, df_user):
    coupon_price_encode_map = {150: 0, 60: 1, 50: 2, 40: 3, 30: 4, 20: 5, 10: 6, 0: 7, float('nan'): 0}
    df_model_input['CouponPrice'] = df_user['CouponPrice'].map(coupon_price_encode_map)


def add_featrues(df_input, df_user_used_coupon):
    encode_region(df_input, df_user_used_coupon)
    encode_device(df_input, df_user_used_coupon)
    version_encode(df_input, df_user_used_coupon)
    account_encode(df_input, df_user_used_coupon)
    time_transform_abs2rela(df_input, df_user_used_coupon)
    encode_baby_birtday(df_input, df_user_used_coupon)
    encode_coupon_price(df_input, df_user_used_coupon)
    df_input['UserCoupon'] = df_user_used_coupon['UseCoupon'].map({True: 1, False: 0})


def build_train_test_dataset(df_input):

    X = df_input.drop(columns=['UserCoupon', 'CouponType'])
    y = df_input['UserCoupon']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def display_training_process(X, y, model):
    train_sizes, train_score, test_score = learning_curve(model,
                                                          X,
                                                          y,
                                                          train_sizes=[i / 20 for i in range(1, 20)],
                                                          cv=10,
                                                          scoring='accuracy')
    train_error = 1 - np.mean(train_score, axis=1)
    test_error = 1 - np.mean(test_score, axis=1)
    plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
    plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
    plt.legend(loc='best')
    plt.xlabel('traing examples')
    plt.ylabel('error')
    plt.show()


def display_roc_auc(y_test, model):
    logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()


df_user_dataset = pd.read_csv('user_data_v1.csv')
df_user_dataset = df_user_dataset.rename(columns={'Unnamed: 0': 'user_id'})
df_input = pd.DataFrame()
df_user_used_coupon = df_user_dataset[~df_user_dataset['UseCoupon'].isnull()]

add_featrues(df_input, df_user_used_coupon)

selected_feature = feature_selection.f_regression(X, y)
X_train, X_test, y_train, y_test = build_train_test_dataset(df_input)
display_training_process(df_input.drop(columns=['UserCoupon', 'CouponType']),
                         df_input['UserCoupon'],
                         LogisticRegression(solver='liblinear'))

model = LogisticRegression(multi_class='ovr', solver='liblinear', n_jobs=1, max_iter=200)
model.fit(X_train, y_train)

print(model.get_params())
print(model.coef_)

y_pred = model.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))

display_training_process(y_test, model)
