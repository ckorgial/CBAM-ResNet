import os
import re
import numpy as np


spectro_dataset_path = ''

lists = [['01', '26'],
         ['02', '10'], ['03'], ['05', '14', '18'], ['06', '15'], ['07'], ['08'],
         ['09'], ['11'], ['13'], ['16'], ['19'], ['20'], ['21'], ['23'], ['24'], ['25'], ['27'], ['28'],
         ['29', '34'], ['30'], ['31'], ['32'], ['33'], ['35']]
lists = {index: value for index, value in enumerate(lists)}

excluded = [4, 12, 17, 22]


X = np.load(spectro_dataset_path + 'X.npy')
y = np.load(spectro_dataset_path + 'y.npy')

# 1) Delete excluded labels
for i in excluded:
    # Find the indices of samples with label 5
    indices_to_delete = np.where(y == i)[0]

    # Delete the samples with label 5
    X = [sample for i, sample in enumerate(X) if i not in indices_to_delete]
    y = np.delete(y, indices_to_delete)


# 2) Assign first label of each sublist
for j in lists.keys():
    correct_label = j
    for l in lists[j]:
        lb = int(l)

        for i in range(len(y)):
            print(y[i], lb)
            if y[i] == lb:
                y[i] = correct_label

# 3) Save
np.save('X.npy', X)
np.save('y.npy', y)



