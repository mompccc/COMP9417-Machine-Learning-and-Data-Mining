import arff
import numpy as np
from sklearn import neighbors

data_file = list(arff.load('ionosphere.arff'))
data = []
classification = []
test = []
test_class = []
k = 5

count = 0
for x in data_file:
    temp = list(x)
    if temp:
        if count < 200:
            classification.append(temp.pop(-1))
            data.append(temp)
            count += 1
        else:
            test_class.append(temp.pop(-1))
            test.append(temp)
data = np.array(data)
test = np.array(test)

print(classification)
print(data)
print(test.shape)

for x in range(1, 10):
    knn = neighbors.KNeighborsClassifier(n_neighbors=x)
    knn.fit(data, classification)

    result = list(knn.predict(test))
    count1 = 0
    for i in range(len(test_class)):
        if test_class[i] != result[i]:
            count1 += 1

    train_result = list(knn.predict(data))
    count2 = 0
    for i in range(len(classification)):
        if classification[i] != train_result[i]:
            count2 += 1

    print("k = {}, train result: {}, test result: {}".format(x, 1-count2/len(train_result), 1-count1/len(result)))