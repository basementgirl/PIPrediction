# -*- coding: utf-8 -*-

import tensorflow as tf
import tempfile
import pandas as pd
import urllib
import numpy as np


import warnings

warnings.filterwarnings("ignore")

# Categorical base columns.


#针对取值较少的类别特征，使用layer.sparse_column_with_keys方法将字符串类型映射到int型。
gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])


#针对不知有多少类别的离散型特征将其映射到"桶"中。该方法将特征中的每一个可能的取值散列分配一个整型ID。
# 如哈希到1000个bucket中，这1000个bucket分别为0~999，那么occupation中的值就会被映射为这1000个中的一个整数
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)


# 将连续型变量直接转换为浮点数。
age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")


#对于分布不平均的连续性变量。对于一些每个段分布密度不均的连续性变量可以做分块。
#这里就是讲age按给定的boundaries分成11个区域，比如样本的age是34，那么输出的就是3，age是21，那么输出的就是1。
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])


#产生交叉特征
wide_columns = [
  gender, native_country, education, occupation, workclass, relationship, age_buckets,
  tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))]



#因为类别型特征无法直接输入到网络中，会报错，所以要进行one—hot编码或者embedding才可以。。
deep_columns = [
  tf.contrib.layers.embedding_column(workclass, dimension=8),
  tf.contrib.layers.embedding_column(education, dimension=8),
  tf.contrib.layers.embedding_column(gender, dimension=8),
  tf.contrib.layers.embedding_column(relationship, dimension=8),
  tf.contrib.layers.embedding_column(native_country, dimension=8),
  tf.contrib.layers.embedding_column(occupation, dimension=8),
  age, education_num, capital_gain, capital_loss, hours_per_week]



model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[50, 25])

# Define the column names for the data sets.
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "gender",
  "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
                        #所属企业、教育水平、婚姻状态、职业、家庭关系、种族、性别、国家

CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]
                        #年龄、教育年限，获利、亏损、工作时长


 #Download the training and test data to temporary files.
# Alternatively, you can download them yourself and change train_file and
# test_file to your own paths.
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)


# Read the training and test data sets into Pandas dataframe.
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)

#print('df_train',df_train)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)


def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.  常数张量
    print('df[k].value',type(df['capital_gain']))
    continuous_cols = {k: tf.constant(df[k].values)       #df[k].value是ndarray； df[k]是Series。continuous_cols是字典
                     for k in CONTINUOUS_COLUMNS}
    print('continuous_cols:',type(continuous_cols))


  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.  稀疏张量
    categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1]) 
      for k in CATEGORICAL_COLUMNS}

    print('categorical_cols',categorical_cols)
    print('items1',type(continuous_cols.items()))
    print('items2',categorical_cols.items())


  # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols, **categorical_cols)
  # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
    return feature_cols, label



def train_input_fn():
    return input_fn(df_train)


def eval_input_fn():
    return input_fn(df_test)


print('df_train shape:', np.array(df_train).shape)
print('df_test shape:', np.array(df_test).shape)

m.fit(input_fn=train_input_fn, steps=20)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
