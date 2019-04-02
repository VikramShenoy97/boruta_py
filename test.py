import sys
import os
import random
import math
import numpy as np
import pandas as pd
from warnings import warn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from scipy import stats
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

Experiment_name = "Spambase Dataset"
file_name = "Spam/Spam_Input"
dataset = pd.read_csv(file_name, header=None)
dataset_array = dataset.values
train_input_test = dataset_array[:,:len(dataset_array[0])-1]
train_output_d_test = dataset_array[:,len(dataset_array[0])-1:]

samples = len(train_input_test)
number_of_features = len(train_input_test[0])

train_input_test = stats.zscore(train_input_test)

stopping_criteria = []

number_of_training_samples = int(samples * 0.70)
test_input = train_input_test[number_of_training_samples:,:]
test_output_d = train_output_d_test[number_of_training_samples:]
number_of_test_samples = len(test_input)

if(number_of_training_samples > 4000):
    limit_train_samples = 4000
else:
    limit_train_samples = number_of_training_samples

train_input = train_input_test[:limit_train_samples,:]
train_output_d = train_output_d_test[:limit_train_samples]

if(number_of_features > 10):
    max_features = 10
else:
    max_features = number_of_features
bo_subset = 99*np.ones((1,max_features))

RF_c = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
xgboost_ensemble = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=200, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=1, seed=2, missing=None)
# XGBoost
boruta = BorutaPy(xgboost_ensemble, n_estimators='auto', verbose=2, random_state=1)
boruta.fit(train_input, np.ravel(train_output_d))
boruta_rank = boruta.ranking_
boruta_count = boruta.n_features_
relevant_indices = boruta_rank.argsort()[:boruta_count]
print relevant_indices
# Random Forest
boruta = BorutaPy(RF_c, n_estimators='auto', verbose=2, random_state=1)
boruta.fit(train_input, np.ravel(train_output_d))
boruta_rank = boruta.ranking_
boruta_count = boruta.n_features_
relevant_indices = boruta_rank.argsort()[:boruta_count]
print relevant_indices
