import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np


def prediction_is_right(prediction, ground_truth):
    i_g = ground_truth.index(1)
    i_p = prediction.index(max(prediction))
    return i_g == i_p

names_ = [str(n) for n in range(1, 1087)]
names_.append("famacha_class")
print("loading dataset...")
data = pd.read_csv("C:\\Users\\fo18103\PycharmProjects\\famatchatable\\training_f_c.data", sep=",",
                   names=names_)
np.random.seed(0)
data = data.sample(frac=1).reset_index(drop=True)
# data = data.truncate(after=150)
print(data)
#
X = data[names_].values
#
y = data["famacha_class"].values.flatten()

s = 400
train_x = X[0:s]
train_y = y[0:s]

test_x = X[s:]
test_y = y[s:]


# we create 40 separable points
# X, y = make_blobs(n_features=3, n_samples=3, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
print("training...")
clf.fit(train_x, train_y)
# d = clf.decision_function(X)
print("predicting...")
predicted = clf.predict(test_x)
# print(X, y)
print(test_y, predicted)

accuracy = accuracy_score(test_y, predicted)
print(accuracy)

# a = X[0:2, 0]
# b = X[2:4, 1]
# plt.scatter(a, b, c=y, s=30, cmap=plt.cm.Paired)
#
# # plot the decision function
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
#
# # create grid to evaluate model
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = clf.decision_function(xy).reshape(XX.shape)
#
# # plot decision boundary and margins
# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])
# # plot support vectors
# ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
#            linewidth=1, facecolors='none', edgecolors='k')
# plt.show()