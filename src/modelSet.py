import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import cross_validation, metrics
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import tempfile
import urllib



def lr_model(X_train, X_test, y_train, y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_y_predict = lr.predict(X_test)
    print(lr_y_predict, y_test)
    print(classification_report(y_test, lr_y_predict, target_names=['0', '1']))



# 模型训练
def xgb_model(X_train, X_test, y_train, y_test):
    xgbc = xgb.XGBClassifier()
    xgbc.fit(X_train, y_train)
    xgbr_y_predict = xgbc.predict(X_test)

    test_auc = metrics.roc_auc_score(y_test, xgbr_y_predict)
    print('auc is :', test_auc)
    print(type(xgbr_y_predict ), type(y_test))
    print(xgbr_y_predict .shape, y_test.shape)
    print(xgbr_y_predict , y_test)
    print(classification_report(y_test, xgbr_y_predict, target_names=['0', '1']))



def map_int(i):
    if i>= 0.5:  # 概率大于0.5预测为1，否则预测为0
        return 1
    else:
        return 0

def ann_model(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(64, input_dim=8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              epochs=20,batch_size=50
              )
    score = model.evaluate(X_test, y_test, batch_size=50)
    predict_y=model.predict(X_test, batch_size=32, verbose=0)

    predict_y=predict_y.reshape(1, -1)     #对预测的概率值变成整数值
    predict_y=pd.Series(predict_y[0])
    predict_y=predict_y.map(map_int)

    res=classification_report(y_test, predict_y, target_names=['0', '1'])
    print(score)
    print(res)



def widedeep_model(X_train, X_test, y_train, y_test):

    model_dir = tempfile.mkdtemp()
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[50, 25])


    m.fit(input_fn=train_input_fn, steps=20)
    results = m.evaluate(input_fn=eval_input_fn, steps=1)