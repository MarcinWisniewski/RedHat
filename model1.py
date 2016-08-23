import os.path
import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


time = datetime.datetime.now()
train_events_path = os.path.expanduser('~/data/kaggle/redhat/act_train.csv')
test_events_path = os.path.expanduser('~/data/kaggle/redhat/act_test.csv')

people_path = os.path.expanduser('~/data/kaggle/redhat/people.csv')

print '...reading files'
train_events = pd.read_csv(train_events_path, usecols=['people_id', 'activity_category',
                                                       'char_1',	'char_2',	'char_3',
                                                       #'char_4',	'char_5', 'char_6',
                                                       'char_7',	'char_8',	'char_9',
                                                       'char_10', 'outcome'
                                                       ])
test_events = pd.read_csv(test_events_path, usecols=['people_id', 'activity_category',
                                                     'char_1',	'char_2',	'char_3',
                                                     #'char_4',	'char_5', 'char_6',
                                                     'char_7',	'char_8',	'char_9',
                                                     'char_10', 'activity_id'
                                                     ])


people = pd.read_csv(people_path, usecols=['people_id', 'group_1', 'char_1', 'char_2', 'char_3',
                                           'char_4', 'char_5', 'char_6', 'char_7',
                                           #'char_8', 'char_9', 'char_10', 'char_11',
                                           #'char_12', 'char_13', 'char_14', 'char_15',
                                           #'char_17', 'char_18', 'char_19', 'char_20',
                                           #'char_21', 'char_22', 'char_23', 'char_24',
                                           #'char_25', 'char_26', 'char_27', 'char_28',
                                           'char_29', 'char_30', 'char_31', 'char_32',
                                           'char_33', 'char_34', 'char_35', 'char_36',
                                           'char_37', 'char_38'])

people.fillna('type -9999', inplace=True)
train_events.fillna('type -9999', inplace=True)
test_events.fillna('type -9999', inplace=True)

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


train_events_people = pd.merge(parsed_train_events, parsed_people, how='left', on='people_id', left_index=True)

classes = train_events['outcome']
features = train_events_people.drop(labels=['outcome', 'people_id'], axis=1)
print '...splitting data'
X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.33, random_state=42)

print '...learning'
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predictions = rf.predict_proba(X_test)
predicted_proba = np.argmax(predictions, axis=1)
fpr, tpr, thresholds = roc_curve(y_test, predicted_proba, pos_label=1)
print 'AUC: %f' % auc(fpr, tpr)


activity_id = test_events['activity_id']
test_events.drop(labels=['activity_id'], axis=1, inplace=True)

print '...parsing test data'
parsed_test_events = pd.DataFrame()
for column in test_events.columns:
    if column == 'people_id':
        parsed_test_events[column] = test_events[column]
    else:
        temp = test_events[column].apply(lambda x: int(x.split()[-1]))
        parsed_test_events[column] = temp


test_events_people = pd.merge(parsed_test_events, parsed_people, how='left', on='people_id', left_index=True)
print '...predicting '
preds = rf.predict_proba(test_events_people.drop(labels=['people_id'], axis=1))
preds = np.argmax(preds, axis=1)

print '...saving submission'
submission = pd.DataFrame()
submission['outcome'] = preds
submission['activity_id'] = activity_id
submission.to_csv('submission' + time.strftime('%m_%d_%H_%M') + '.csv', index=False, float_format='%.3f')

