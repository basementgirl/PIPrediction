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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score



def lr_model(X_train, X_test, y_train, y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_y_predict = lr.predict(X_test)
    lr_y_predict_proba = lr.predict_proba(X_test)
    print('lr_y_predict',lr_y_predict, 'lr_y_predict_proba',lr_y_predict_proba)


    print('accu',accuracy_score(lr_y_predict,y_test))
    test_auc = metrics.roc_auc_score(y_test, lr_y_predict_proba[:,1])  # 验证集上的auc值
    print('test_auc', test_auc)
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

    test_auc = metrics.roc_auc_score(y_test, xgbr_y_predict)  # 验证集上的auc值
    print('test_auc', test_auc)
    print(classification_report(y_test, xgbr_y_predict, target_names=['0', '1']))


def gbdt_model(X_train, X_test, y_train, y_test):
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    gbr_y_predict = gbc.predict(X_test)
    gbr_y_predict_proba = gbc.predict_proba(X_test)

    #print(gbr_y_predict , y_test)

    print('accu',accuracy_score(gbr_y_predict,y_test))
    test_auc = metrics.roc_auc_score(y_test, gbr_y_predict_proba[:,1] )  # 验证集上的auc值
    print('test_auc', test_auc)
    print(classification_report(y_test, gbr_y_predict, target_names=['0', '1']))



def map_int(i):
    if i>= 0.5:  # 概率大于0.5预测为1，否则预测为0
        return 1
    else:
        return 0


def ann_model(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(64, input_dim=8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(28, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    history = LossHistory()
    model.fit(X_train, y_train,
              epochs=50,batch_size=28,validation_split=0.4,verbose=0, callbacks=[history])

    #print('history.losses',history.losses)

    print('111',history.loss_plot('epoch'))


    score = model.evaluate(X_test, y_test, batch_size=50)
    predict_y=model.predict(X_test, verbose=0)

    predict_y=predict_y.reshape(1, -1)     #对预测的概率值变成整数值
    predict_y=pd.Series(predict_y[0])

    print('predict_y',predict_y)
    predict_y2=predict_y.map(map_int)

    res=classification_report(y_test, predict_y2, target_names=['0', '1'])

    print('accu',accuracy_score(predict_y2,y_test))
    test_auc = metrics.roc_auc_score(y_test, predict_y)  # 验证集上的auc值
    print('test_auc',test_auc)

    print('score',score)
    print('res',res)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc ['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc ['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc [loss_type], 'b', label='val_acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val_loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig('loss_with_epoch')





