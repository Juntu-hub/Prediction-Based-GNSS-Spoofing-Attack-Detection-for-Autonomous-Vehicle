import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
def average(seq, total=0.0):
  num = 0
  for item in seq:
    total += item
    num += 1
  return total / num

if __name__ == '__main__':

    CSV_FILE_PATH = 'D:\\comma2k19\\Chunk_03\\99c94dc769b5d96e_2018-05-01--08-13-53.csv'
    df = pd.read_csv(CSV_FILE_PATH)
    values = df.to_numpy()
    times = values[:, -1]
    distance = values[:, -2]
    model = tf.keras.models.load_model('lstm.model')
    test_X = values[:, :3]
    # 因为训练的时候输入特征是归一化的，所以预测的时候也要将输入特征归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    test_X = scaler.fit_transform(test_X)
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # train_len = (int)(0.75 * len(values[:, 0]))
    # train = values[:train_len, :]
    # test = values[train_len:, :]
    test_y = distance
    yhat = model.predict(test_X)[:, 0]
    rmse = math.sqrt(mean_squared_error(yhat, test_y))
    print('Test RMSE: %.3f' % rmse)
    scores = model.evaluate(test_X, test_y)
    rmse = math.sqrt(mean_squared_error(yhat, test_y))
    plt.plot(times, yhat, label='prediction')
    plt.plot(times, distance, label="ground_truth")
    plt.title('Comparison between truth and prediction', fontsize=18)
    plt.xlabel('Boot time (s)', fontsize=18)
    plt.ylabel('Distance travelled during single timestamp (m) ', fontsize=12)
    plt.legend()
    plt.show()
    min = min((distance - yhat), key=abs)
    max = max((distance - yhat), key=abs)
    avr = average(distance-yhat)
    print('Min:%f' % min)
    print('Max:%f' % max)
    print('average:%f' % avr)
