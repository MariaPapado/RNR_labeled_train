import numpy as np

l1 = [1,2,3,4]

l2=[ [1,2,3,4], [1,2,3,4], [1,2,3,4]  ]


for l in range(0, len(l2)):
    soft_pred = l2[l]
    print('a')
    print(l1)
    for t in range(len(l1)):
        l1[t] +=soft_pred[t]
    print(l1)

l1 = [t/len(l2) for t in l1]
print(l1)
