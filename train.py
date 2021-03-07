import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras import optimizers
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



if __name__ == '__main__':

    train_CSV_FILE_PATH = 'D:\\comma2k19\\Chunk_01\\b0c9d2329ad1606b_2018-08-02--08-34-47.csv'
    test_CSV_FILE_PATH = 'D:\\comma2k19\\Chunk_01\\b0c9d2329ad1606b_2018-08-01--21-13-49.csv'
    train_df = pd.read_csv(train_CSV_FILE_PATH)
    test_df = pd.read_csv(test_CSV_FILE_PATH)
    train_values = train_df.to_numpy()
    train_times = train_values[:, -1]
    train_distance = train_values[:, -2]
    test_values = test_df.to_numpy()
    test_times = test_values[:, -1]
    test_distance = test_values[:, -2]
    # 将输入特征归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X, train_y = scaler.fit_transform(train_values[:, :-2]), train_distance
    test_X, test_y = scaler.fit_transform(test_values[:, :-2]), test_distance
    # # 将四分之三作为训练集
    # train_len = len(times)
    # train = values[:train_len, :]
    # test = values[train_len:, :]
    # 划分输入（CAN_speed,steering_angel, acceleration_forward）输出（distance)
    # train_X, train_y = train, distance[:train_len]
    # test_X, test_y = test, distance[train_len:]
    # 将输入（X）改造为LSTM的输入格式，即[samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # 设计网络
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    # 设置学习率等参数
    # adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=50, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    model.save('lstm.model')
    # full_X = values[:, :3]
    # full_X = full_X.reshape((full_X.shape[0], 1, full_X.shape[1]))
    train_yhat = model.predict(train_X)[:, 0]
    test_yhat = model.predict(test_X)[:, 0]
    rmse = math.sqrt(mean_squared_error(test_yhat, test_y))
    print('Test RMSE: %.3f' % rmse)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    # plt.plot(times, yhat, label='prediction')
    # plt.plot(times, distance, label="ground_truth")
    # plt.title('Comparison between truth and prediction', fontsize=18)
    # plt.xlabel('Boot time (s)', fontsize=18)
    # plt.ylabel('Distance travelled during single timestamp (m) ', fontsize=12)
    plt.legend()
    plt.show()