import numpy as np
import pandas as pd
#import xgboost as xgb
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import cross_validation, metrics
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import tempfile
import urllib
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.callbacks import ModelCheckpoint
import keras
import matplotlib.pyplot as plt
from hyperopt import hp
from hyperopt import hp, fmin, rand, tpe, space_eval
from featureEngineeringClassify import get_comp_train_and_test



X_train, X_test, y_train, y_test=get_comp_train_and_test()

space={"node_nums_1":hp.randint('node_nums_1', 64),
        "node_nums_2":hp.randint('node_nums_2', 32),
       "epoch_nums": hp.randint('epoch_nums', 10),
       "batch_size_num": hp.randint('batch_size_num', 50),
       "drop_out1":hp.uniform("drop_out1",0,0.5),
       "drop_out2":hp.uniform("drop_out2",0,0.3)
        }


def best_ann_model(argsDict):
    global X_train, X_test, y_train, y_test
    model = Sequential()
    print('111')

    model.add(Dense(argsDict["node_nums_1"]+1, input_dim=8, activation='relu'))
    model.add(Dropout(argsDict["drop_out1"]))
    model.add(Dense(argsDict["node_nums_2"]+1, activation='relu'))
    model.add(Dropout(argsDict["drop_out2"]))
    model.add(Dense(1, activation='sigmoid'))
    print('hhh')

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              epochs=argsDict["epoch_nums"]+1,batch_size=argsDict["batch_size_num"]+1,verbose=0)
    print('eee')

    predict_y=model.predict(X_test, verbose=0)
    predict_y=predict_y.reshape(1, -1)
    predict_y=pd.Series(predict_y[0])
    print('predict_y',predict_y)

    test_auc = metrics.roc_auc_score(y_test, predict_y)  # 验证集上的auc值
    return -test_auc


best = fmin(best_ann_model, space, algo=rand.suggest,max_evals=100)
print(best)
