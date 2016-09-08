import os.path
import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import auc, roc_curve
import xgboost as xgb

import matplotlib.pyplot as plt

time = datetime.datetime.now()
params = {}
params['objective'] = 'binary:logistic'
params['booster'] = 'gblinear'
params['eval_metric'] = 'auc'
params['eta'] = 0.02
params['learning_rate'] = 0.05
params['max_depth'] = 3
params['subsample'] = 0.7
params['colsample_bytree'] = 0.7
params['verbose'] = 0

train_events_path = os.path.expanduser('~/data/kaggle/redhat/act_train.csv')
test_events_path = os.path.expanduser('~/data/kaggle/redhat/act_test.csv')

people_path = os.path.expanduser('~/data/kaggle/redhat/people.csv')

print '...reading files'
train_events = pd.read_csv(train_events_path, usecols=['people_id', 'date', 'activity_category',
                                                       'char_1',	'char_2',	'char_3',
                                                       'char_4',	'char_5', 'char_6',
                                                       'char_7',	'char_8',	'char_9',
                                                       'char_10',
                                                       'outcome'
                                                       ], parse_dates=['date'])
test_events = pd.read_csv(test_events_path, usecols=['people_id', 'date', 'activity_category',
                                                     'char_1',	'char_2',	'char_3',
                                                     'char_4',	'char_5', 'char_6',
                                                     'char_7',	'char_8',	'char_9',
                                                     'char_10',
                                                     'activity_id'
                                                     ], parse_dates=['date'])


people = pd.read_csv(people_path, usecols=['people_id', 'date', 'group_1', 'char_1', 'char_2', 'char_3',
                                           'char_4', 'char_5', 'char_6', 'char_7',
                                           'char_8', 'char_9', 'char_10', 'char_11',
                                           'char_12', 'char_13', 'char_14', 'char_15',
                                           'char_17', 'char_18', 'char_19', 'char_20',
                                           'char_21', 'char_22', 'char_23', 'char_24',
                                           'char_25', 'char_26', 'char_27', 'char_28',
                                           'char_29', 'char_30', 'char_31', 'char_32',
                                           'char_33', 'char_34', 'char_35', 'char_36',
                                           'char_37', 'char_38'], parse_dates=['date'])


people.fillna('type 999999', inplace=True)
train_events.fillna('type 999999', inplace=True)
test_events.fillna('type 999999', inplace=True)

train_events_date = train_events['date'].dt
test_events_date = test_events['date'].dt
people_date = people['date'].dt

train_events.drop(labels=['date'], axis=1, inplace=True)
test_events.drop(labels=['date'], axis=1, inplace=True)
people.drop(labels=['date'], axis=1, inplace=True)

activity_id = test_events['activity_id']
test_events.drop(labels=['activity_id'], axis=1, inplace=True)

print '...parsing train data'
parsed_train_events = pd.DataFrame()
for column in train_events.columns:
    if column == 'people_id' or column == 'outcome':
        parsed_train_events[column] = train_events[column]
    else:
        temp = train_events[column].apply(lambda x: int(x.split()[-1]))
        parsed_train_events[column] = temp

parsed_people = pd.DataFrame()
for column in people.columns:
    if 'group_1' in column or len(column) == 6:
        temp = people[column].apply(lambda x: int(x.split()[-1]))
        parsed_people[column] = temp
    else:
        parsed_people[column] = people[column]

print '...parsing test data'
parsed_test_events = pd.DataFrame()
for column in test_events.columns:
    if column == 'people_id':
        parsed_test_events[column] = test_events[column]
    else:
        temp = test_events[column].apply(lambda x: int(x.split()[-1]))
        parsed_test_events[column] = temp

train_events['year'] = train_events_date.year
train_events['month'] = train_events_date.month
train_events['day'] = train_events_date.day
train_events['weekend'] = (train_events_date.weekday >= 5).astype(int)


people['year'] = people_date.year
people['month'] = people_date.month
people['day'] = people_date.day
people['weekend'] = (people_date.weekday >= 5).astype(int)

test_events['year'] = test_events_date.year
test_events['month'] = test_events_date.month
test_events['day'] = test_events_date.day
test_events['weekend'] = (test_events_date.weekday >= 5).astype(int)


train_events_people = pd.merge(parsed_train_events, parsed_people, how='left', on='people_id', left_index=True)
test_events_people = pd.merge(parsed_test_events, parsed_people, how='left', on='people_id', left_index=True)

whole_events_people = pd.concat([train_events_people.drop(labels=['outcome', 'people_id'], axis=1),
                                 test_events_people.drop(labels=['people_id'], axis=1)])

ohe = OneHotEncoder(categorical_features=range(len(list(whole_events_people))-1))
ohe.fit(whole_events_people)

classes = train_events['outcome']
features = ohe.transform(train_events_people.drop(labels=['outcome', 'people_id'], axis=1))
print '...splitting data'
X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.33, random_state=42)

print '...learning'
d_train = xgb.DMatrix(X_train, label=y_train, silent=True)
d_valid = xgb.DMatrix(X_test, label=y_test, silent=True)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 50000, evals=watchlist, early_stopping_rounds=20)
predictions = clf.predict(xgb.DMatrix(X_test))
predicted_proba = predictions
fpr, tpr, thresholds = roc_curve(y_test, predicted_proba, pos_label=1)
print 'AUC: %f' % auc(fpr, tpr)

d_test = xgb.DMatrix(ohe.transform(test_events_people.drop(labels=['people_id'], axis=1)))
print '...predicting '
preds = clf.predict(d_test)

print '...saving submission'
submission = pd.DataFrame()
submission['outcome'] = preds
submission['activity_id'] = activity_id
submission.to_csv('submission_' + time.strftime('%m_%d_%H_%M') + '_' + '{0:.5f}'.format(auc(fpr, tpr)) + '.csv',
                  index=False, float_format='%.3f')
