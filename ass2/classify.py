import re
import numpy as np
import matplotlib.pyplot as plt
import math


def EUC(train_data, train_class, test_data, k):  # use Euclidean-Distance to compute distance
    instances = train_data.shape[0]
    minus = np.tile(test_data, (instances, 1)) - train_data  # step 1: minus
    squared_m = minus ** 2  # step 2: square
    squared_dist = squared_m.sum(axis=1)  # step 3: sum up
    distance = squared_dist ** 0.5  # step 4: extraction of a root

    sorted_index = np.argsort(distance)  # step 5: sort the distance in ascending and return rank list
    class_count = {}  # dict to store 'g' and 'b' count
    for i in range(k):
        temp_class = train_class[sorted_index[i]]
        class_count[temp_class] = class_count.get(temp_class, 0) + 1  # this line is non-weighted
        #class_count[temp_class] = class_count.get(temp_class, 0) + 1*(math.e**(-distance[sorted_index[i]]**2 /(2*0.3**2)))  # this line is weighted using Gaussian Function
    max_count = 0
    for key, value in class_count.items():
        if value >= max_count:
            max_count = value
            max_index = key
    return max_index


def cross_validation(file, K, L, C):
    result = []
    for n in range(L):  # repeat L times and get each line tested
        train_data = []
        train_class = []
        temp_list = list(file[n])
        test_data = temp_list[:-1]
        for i in range(L):
            if i != n:  # set all lines except the test line into training data
                temp_list = list(file[i])
                train_class.append(temp_list.pop(-1))
                train_data.append(temp_list)
        train_data = np.array(train_data)
        result.append(EUC(train_data, train_class, test_data, K))  # get all classifications into one list
    count1 = 0
    for i in range(L):
        if result[i] != C[i]:
            count1 += 1  # count wrong classifications
    return 1 - count1/len(C)


def classify(file, K):
    train_data = []
    train_class = []
    test_data = []
    test_class = []
    count2 = 0
    for xx in file:
        the_list = list(xx)
        if the_list:
            if count2 < 200:  # use the first 200 lines as training data
                train_class.append(the_list.pop(-1))
                train_data.append(the_list)
                count2 += 1
            else:  # the rest is test data
                test_class.append(the_list.pop(-1))
                test_data.append(the_list)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    result = []
    for d in test_data:
        result.append(EUC(train_data, train_class, d, K))
    count1 = 0
    for i in range(len(result)):
        if result[i] != test_class[i]:
            count1 += 1  # count wrong classifications

    result1 = []
    for d in train_data:
        result1.append(EUC(train_data, train_class, d, K))
    count2 = 0
    for i in range(len(result1)):
        if result1[i] != train_class[i]:
            count2 += 1  # count wrong classifications
    return 1 - count1/len(test_class), 1 - count2/len(train_class)  # training data result and test data result


# ----------------------------------------------------------
data_file = []
for line in open('ionosphere.arff'):  # read file into list
    line = line.strip()
    if not re.match(r'\%',line):
        if not re.match(r'\@', line) and line:
            temp = line.split(",")
            temp_array = []
            for i in range(0, len(temp)):
                if i == len(temp) - 1:
                    temp_array.append(temp[i])
                else:
                    temp_array.append(float(temp[i]))
            data_file.append(temp_array)

length = len(data_file)
classification = []
x_aix = []
y_aix0 = []
y_aix1 = []

count = 0
for x in data_file:
    temp = list(x)
    classification.append(temp.pop(-1))

for k in range(1, 30):
    result = cross_validation(data_file, k, 200, classification)
    y_aix0.append(result)
    print("cross-validation: k={}, {}".format(k, result))

for k in range(1, 30):
    result = classify(data_file, k)
    x_aix.append(k)
    y_aix1.append(result[0])
    print("k={}, train result: {}, test result: {}".format(k, result[1], result[0]))


plt.plot(x_aix, y_aix0, 'rp:')
plt.plot(x_aix, y_aix1, 'bp--')
plt.ylabel('Accuracy  Rate %')
plt.xlabel('K  value')
plt.title('Results  of  Classification')
plt.show()