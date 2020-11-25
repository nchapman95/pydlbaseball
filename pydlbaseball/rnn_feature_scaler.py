import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_create_split(dataset_train, test_set,time_steps = 20):

  training_set = dataset_train.iloc[:, 1:2].values[:237]
  sc = MinMaxScaler(feature_range = (0, 1))
  training_set_scaled = sc.fit_transform(training_set)

  X_train = []
  y_train = []
  for i in range(time_steps, 237):
      X_train.append(training_set_scaled[i-time_steps:i, 0])
      y_train.append(training_set_scaled[i, 0])
  X_train, y_train = np.array(X_train), np.array(y_train)

  ## reshaping
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


  dataset_total = dataset_train['cases']
  inputs = dataset_total[len(dataset_total) - len(test_set) - time_steps:].values
  inputs = inputs.reshape(-1,1)
  inputs = sc.transform(inputs)

  X_test = []
  y_test = []
  for i in range(time_steps, time_steps + len(test_set)):
      X_test.append(inputs[i-time_steps:i, 0])
      y_test.append(inputs[i,0])
  X_test, y_test = np.array(X_test), np.array(y_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

  return X_train, X_test, y_train, y_test, sc 
