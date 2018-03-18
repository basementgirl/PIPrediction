
# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
#x = Embedding(output_dim=16, input_dim=10000, input_length=100)(main_input)
import numpy as np
from keras.layers.embeddings import  Embedding
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout



model = Sequential()
model.add(Embedding(1000, output_dim=64, input_length=10))  # input_dim=1000是字典长度
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# model.input_shape = (None,10), where None is the batch dimension, 10 is input_length(time_step), input_dim is 1.
# model.output_shape = (None, 10, 64), where None is the batch dimension, 10 is output_length, 64 is output_dim.

# 32 samples, 10 time steps, input is a number between 0 and 1000 in 2-D array input_array.
input_array = np.random.randint(1000, size=(32, 10))
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)

'''
def built_feature_and_flag(self, data, period=52):
    i = len(data) - 1 - period
    feature_set = []
    flag_set = []
    while i >= 0:
        feature_set.append(data[i:i + period])  # 修复了大bug啊。
        flag_set.append(data[i + period])
        i -= 1
    return np.array(feature_set), np.array(flag_set)


def built_train_and_test_set(self, feature_set, flag_set):
    # next_priod=20     #可修改该值，来确定测试集的大小
    next_priod = int(len(feature_set) * 0.2)

    train_x, test_x = feature_set[:-next_priod], feature_set[-next_priod:]
    train_y, test_y = flag_set[:-next_priod], flag_set[-next_priod:]

    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    # LSTM数据的输入形式：（samples,timesteps,input_dim）input_dim数据的表示形式的维度，timestep则为总的时间步数？？？？？？？？？？？？？？
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    return train_x, train_y, test_x, test_y

def lstm_prediction(self,trainX, trainY, testX,test_y,look_back=52):
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, look_back)))
    #model.add(Dropout(0.4))

    #model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.4))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

    model.fit(trainX, trainY, epochs=60,batch_size=1,validation_split=0.33,verbose=2)

    #validation_split = 0.33,
    trainPredict=model.predict(trainX)
    testPredict = model.predict(testX)


    return trainPredict, testPredict'''