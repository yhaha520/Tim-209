from sklearn import datasets
import numpy as np
import collections
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


irirs = datasets.load_iris()
irirs_data = irirs.data
irirs_label = irirs.target

size = len(irirs_data)

random_irirs_data = np.empty(irirs_data.shape, dtype=irirs_data.dtype)
random_irirs_label = np.empty(irirs_label.shape, dtype=irirs_label.dtype)

index = np.random.permutation(np.arange(size))

# random shuffle the origin dataset to cross-validation
for i in range(len(index)):
    random_irirs_data[i] = irirs_data[index[i]]
    random_irirs_label[i] = irirs_label[index[i]]



k = 5
intervals = int(size/k)
knn_accuracy, svm_accuracy, DecisionTree_accuracy = 0, 0, 0

for i in range(1, k+1):
    # print("now intervals:", (i-1)*intervals, "to",  i*intervals)
    x_test, y_test = random_irirs_data[(i-1)*intervals:i*intervals], random_irirs_label[(i-1)*intervals:i*intervals]
    x_train = np.concatenate((random_irirs_data[0:(i-1)*intervals], random_irirs_data[i*intervals:size]), axis=0)
    y_train = np.concatenate((random_irirs_label[0:(i-1)*intervals], random_irirs_label[i*intervals:size]), axis=0)

    knn = KNeighborsClassifier(n_neighbors=3, weights = 'uniform', algorithm='auto', leaf_size = 30, p=2,
                               metric = 'minkowski', metric_params=None)
    knn.fit(x_train, y_train)

    knn_result = collections.Counter(knn.predict(x_test) - y_test)
    knn_accuracy += knn_result[0]/intervals

    svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto',
              kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
    svm.fit(x_train, y_train)

    svm_result = collections.Counter(svm.predict(x_test) - y_test)
    svm_accuracy += svm_result[0]/intervals

    DecisionTree = DecisionTreeClassifier()
    DecisionTree.fit(x_train, y_train)

    DecisionTree_result = collections.Counter(DecisionTree.predict(x_test) - y_test)
    DecisionTree_accuracy += DecisionTree_result[0]/intervals

print(knn_accuracy/k)
print(svm_accuracy/k)
print(DecisionTree_accuracy/k)

