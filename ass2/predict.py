import numpy as np
import matplotlib.pyplot as plt
import re


def EUC1(train_data, train_class, test_data, k):  # this is weighted prediction
    instances = train_data.shape[0]
    minus = np.tile(test_data, (instances, 1)) - train_data  # step 1: minus
    squared_m = minus ** 2  # step 2: square
    squared_dist = squared_m.sum(axis=1)  # step 3: sum up
    distance = squared_dist ** 0.5  # step 4: extraction of a root

    sorted_index = np.argsort(distance)  # step 5: sort the distance in ascending and return rank list
    all_price = []  # list to store all nearest k prices
    D = distance[sorted_index[0]]+1  # set the distance of first point as basement
    denominator = 0
    for i in range(k):
        ratio = D/(distance[sorted_index[i]]+1)  # compute the ratio of current point distance and basement point distance
        temp_price = train_class[sorted_index[i]] * ratio  # plus weighted price
        all_price.append(temp_price)
        denominator += ratio
    return sum(all_price)/denominator


def EUC(train_data, train_class, test_data, k):  # this is non-weighted prediction
    instances = train_data.shape[0]
    minus = np.tile(test_data, (instances, 1)) - train_data
    squared_m = minus ** 2
    squared_dist = squared_m.sum(axis=1)
    distance = squared_dist ** 0.5
    sorted_index = np.argsort(distance)
    all_price = []
    for i in range(k):
        temp_price = train_class[sorted_index[i]]
        all_price.append(temp_price)
    return sum(all_price)/len(all_price)  # simply compute average price of k nearest points


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
        result.append(EUC(train_data, train_class, test_data, K))  # change EUC to EUC1 to implement weighted kNN
    out = []  # a list to store all error rate
    temp_minus = []  # a list to store D-value for later root-mean-square error computation
    for i in range(L):
        if result[i] < C[i]:
            temp_minus.append(C[i]-result[i])  # root-mean-square error computation step 1: D-value
            per = (C[i]-result[i])/C[i]  # error rate computation step 1
        else:
            temp_minus.append(result[i] - C[i])
            per = (result[i] - C[i])/C[i]
        out.append(per)
    sum_per = sum(out)/len(out) * 100  # get average error rate

    A = 0
    for r in temp_minus:
        A += r**2  # root-mean-square error computation step 2: square
    V = (A/len(temp_minus))**0.5  # root-mean-square error computation step 3: extraction of a root
    return sum_per, V  # average error rate, root-mean-square error


# ----------------------------------------------------------
data_file = []
for line in open('autos.arff'):  # read file into list
    line = line.strip()
    if not re.match(r'\%',line):
        if not re.match(r'.*\?',line):
            if not re.match(r'\@', line) and line:
                temp = line.split(",")
                temp_array = []
                price = ''
                for i in range(0, len(temp)):
                    if not re.match(r'.*[A-Z,a-z]+', temp[i]):
                        if i == len(temp) - 2:
                            price = temp[i]
                        elif i == len(temp) - 1:
                            temp_array.append(float(temp[i]))
                            temp_array.append(float(price))
                        else:
                            temp_array.append(float(temp[i]))
                data_file.append(temp_array)
print(data_file)

length = len(data_file)
classification = []

count = 0
for x in data_file:
    temp = list(x)
    classification.append(temp.pop(-1))

#print(classification)
x_aix = []
y_aix0 = []
y_aix1 = []
for k in range(1, 40):
    x_aix.append(k)
    result = cross_validation(data_file, k, length, classification)
    y_aix0.append(result[0])
    y_aix1.append(result[1])
    print("cross-validation: k={}, error={}, RMSE={}".format(k, result[0], result[1]))
fig,picture1 = plt.subplots()
picture2 = picture1.twinx()
picture1.plot(x_aix, y_aix0, 'bp--')
picture1.set_ylabel('Average  Deviation  Error  Rate %')
picture1.set_xlabel('K  value')
picture1.set_title('Results  of  Cross-Validation')
picture2.plot(x_aix, y_aix1, 'rp:')
picture2.set_ylabel("RMSE  for  each  K")
plt.show()
