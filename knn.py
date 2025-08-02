import numpy as np
from collections import Counter

def euclidean_distance(a,b):
    return np.sqrt(np.sum((np.array(a)-np.array(b)))**2)

def knn_predict(training_data,training_label,test_data,k):
    distances=[]
    for i in range(len(training_data)):
      distance=euclidean_distance(test_data,training_data[i])
      distances.append((distance,training_label[i])) #this changes it in tuple
    distances.sort(key=lambda x: x[0])  #using lambda function for define an unnamed function 
    # def my_function(x):
    #   return x[0]
    k_nearest=[label for _, label in distances[:k]] 
    return Counter(k_nearest).most_common(1)[0][0]

training_data = [[2, 4], [2, 6], [6, 4], [5, 7], [7, 8]]
training_label = ['B', 'A', 'B', 'B', 'A']
test_data = [7, 1]
k = 3

prediction=knn_predict(training_data,training_label,test_data,k)
print(prediction)

