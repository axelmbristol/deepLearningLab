import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

N_INPUT_DATA = 5 #features
MINS_IN_A_DAY = 1440
N_DAYS = 6
# count = int([line.rstrip('\n') for line in open("C:\\Users\\fo18103\PycharmProjects\\famatchatable\\count.data")][0]) * N_INPUT_DATA
count = int(MINS_IN_A_DAY*N_DAYS)*N_INPUT_DATA
print("features dimension is %d." % count)

names_ = [str(n) for n in range(1, count)]
names_.append("famacha_class")
print("loading dataset...")
data_frame = pd.read_csv("C:\\Users\\fo18103\PycharmProjects\\training_data_generator\\src\\training_time_domain.data", sep=",",
                         names=names_)
np.random.seed(0)
data_frame = data_frame.sample(frac=1).reset_index(drop=True)
print(data_frame)

data_frame = data_frame.fillna(-1)

# data_frame = data_frame.interpolate(limit_direction='both')

print(data_frame.isnull().values.any())
print(data_frame)

X = data_frame[names_].values
y = data_frame["famacha_class"].values.flatten()

s = 100
train_x = X[0:s]
train_y = y[0:s]

test_x = X[s:]
test_y = y[s:]

clf = svm.SVC(kernel='rbf', C=1000)

print("training...")
clf.fit(train_x, train_y)
print("predicting...")
predicted = clf.predict(test_x)
print(test_y, predicted)
accuracy = accuracy_score(test_y, predicted)
print(accuracy)

