from sklearn import datasets
import numpy as np
from sklearn import svm

irirs = datasets.load_iris()

irirs_data = irirs.data
irirs_label = irirs.target

random_irirs_data = np.empty(irirs_data.shape, dtype=irirs_data.dtype)
random_irirs_label = np.empty(irirs_label.shape, dtype=irirs_label.dtype)

index = np.random.permutation(np.arange(len(irirs_data)))

# random shuffle the origin dataset to cross-validation
for i in range(len(index)):
    # print(irirs_data[index[i]], irirs_label[index[i]])
    random_irirs_data[i] = irirs_data[index[i]]
    random_irirs_label[i] = irirs_label[index[i]]

x_train, x_test = random_irirs_data[:120], random_irirs_data[120:]
y_train, y_test = random_irirs_label[:120], random_irirs_label[120:]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
           weights='uniform')

result = knn.predict(x_test)
# print(knn.predict(x_test))
print(y_test)


