import numpy as np
from collections import Counter

# my k-NN algorithms
def knn(features_train, label_train, features_test, k, distance_metric):
    label_pred = []
    for test_sample in features_test:
        distances = [distance_metric(test_sample, features_train) for features_train in features_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [label_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        label_pred.append(most_common)
    return np.array(label_pred)
