### 主要就三个文件：

1. modelSet  ：写了几个模型，方便对比。被featureEngineeringClassify调用。
2. builtSession ： 建立购买周期。被featureEngineeringClassify调用。
3. featureEngineeringClassify： 调用上述模块。


ps：用啥模型，在modelSet中调啥模型。在featureEngineeringClassify的第五行，改下from modelSet import ann_model
和第128行， print(ann_model(X_train, X_test, y_train, y_test))改下模型名。

